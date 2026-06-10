"""
run_trellis.py — Trellis 2 production runner
  - Auto run-ID + JSON + plaintext log per run
  - Per-step resource snapshots (RAM/Swap/VRAM/CPU/elapsed)
  - Watchdog: kills run if any step hangs > WATCHDOG_TIMEOUT seconds
  - Auto-SCP result GLB to laptop via Tailscale on success

Usage:
    python run_trellis.py assets/example_image/T.png
    python run_trellis.py assets/example_image/T.png --texture-size 2048 --no-push
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
os.environ["HSA_XNACK"] = "1"

import argparse
import gc
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import psutil
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
import o_voxel


# ── Config ────────────────────────────────────────────────────────────────────

LAPTOP_TAILSCALE    = "root@100.74.177.32"
LAPTOP_OUTPUT_DIR   = "/mnt/d/GameDev/astroforge/Output"
RUNS_DIR            = Path("runs")
WATCHDOG_TIMEOUT    = 300   # seconds per step before watchdog fires
ENVMAP_PATH         = "assets/hdri/forest.exr"


# ── Run logger ────────────────────────────────────────────────────────────────

class RunLogger:
    def __init__(self, run_id: str, params: dict):
        self.run_id   = run_id
        self.run_dir  = RUNS_DIR / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path  = self.run_dir / "run.log"
        self.json_path = self.run_dir / "run.json"
        self.t0        = time.time()
        self.snapshots = []
        self.data = {
            "run_id":     run_id,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "params":     params,
            "snapshots":  self.snapshots,
            "result":     None,
        }
        self._log_file = open(self.log_path, "w", buffering=1)
        self._flush_json()
        self.print(f"=== Run {run_id} started ===")
        self.print(f"Params: {json.dumps(params, indent=2)}")

    def print(self, *args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self._log_file.write(msg + "\n")

    def snapshot(self, label: str):
        vm  = psutil.virtual_memory()
        sw  = psutil.swap_memory()
        cpu = psutil.cpu_percent(interval=0.3)
        va  = torch.cuda.memory_allocated() / 1e9
        vr  = torch.cuda.memory_reserved()  / 1e9
        vt  = torch.cuda.get_device_properties(0).total_memory / 1e9
        elapsed = time.time() - self.t0

        snap = {
            "label":        label,
            "elapsed_s":    round(elapsed, 1),
            "cpu_pct":      round(cpu, 1),
            "ram_used_gb":  round(vm.used / 1e9, 2),
            "ram_total_gb": round(vm.total / 1e9, 2),
            "ram_pct":      round(vm.percent, 1),
            "swap_used_gb": round(sw.used / 1e9, 2),
            "swap_pct":     round(sw.percent, 1),
            "vram_alloc_gb":  round(va, 2),
            "vram_reserv_gb": round(vr, 2),
            "vram_total_gb":  round(vt, 2),
        }
        self.snapshots.append(snap)
        self._flush_json()

        self.print(
            f"\n{'='*62}\n"
            f"  {label}\n"
            f"{'='*62}\n"
            f"  Elapsed   : {elapsed:7.1f}s\n"
            f"  CPU       : {cpu:5.1f}%\n"
            f"  RAM       : {vm.used/1e9:5.2f} / {vm.total/1e9:.2f} GB  ({vm.percent:.1f}%)\n"
            f"  Swap      : {sw.used/1e9:5.2f} / {sw.total/1e9:.2f} GB  ({sw.percent:.1f}%)\n"
            f"  VRAM alloc: {va:5.2f} GB\n"
            f"  VRAM reserv:{vr:5.2f} / {vt:.2f} GB\n"
        )

    def finish(self, success: bool, output_path: str = None, error: str = None):
        elapsed = time.time() - self.t0
        self.data["result"] = {
            "success":    success,
            "elapsed_s":  round(elapsed, 1),
            "output":     output_path,
            "error":      error,
            "ended_at":   datetime.utcnow().isoformat() + "Z",
        }
        self._flush_json()
        status = "SUCCESS" if success else "FAILED"
        self.print(f"\n=== Run {self.run_id} {status} in {elapsed:.1f}s ===")
        if error:
            self.print(f"Error: {error}")
        self._log_file.close()

    def _flush_json(self):
        self.json_path.write_text(json.dumps(self.data, indent=2))


# ── Watchdog ──────────────────────────────────────────────────────────────────

class Watchdog:
    def __init__(self, timeout: int, logger: RunLogger):
        self._timeout = timeout
        self._logger  = logger
        self._reset_time = time.time()
        self._label   = "init"
        self._stopped = False
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ping(self, label: str):
        self._label = label
        self._reset_time = time.time()

    def stop(self):
        self._stopped = True

    def _run(self):
        while not self._stopped:
            time.sleep(5)
            if time.time() - self._reset_time > self._timeout:
                self._logger.print(
                    f"\n[WATCHDOG] Step '{self._label}' exceeded {self._timeout}s — killing run!"
                )
                self._logger.finish(success=False, error=f"Watchdog timeout in step: {self._label}")
                os.kill(os.getpid(), signal.SIGTERM)


# ── SCP push ─────────────────────────────────────────────────────────────────

def push_to_laptop(local_path: str, run_id: str, logger: RunLogger) -> bool:
    remote_path = f"{LAPTOP_TAILSCALE}:{LAPTOP_OUTPUT_DIR}/{run_id}.glb"
    logger.print(f"[PUSH] Copying to laptop: {remote_path}")
    try:
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", LAPTOP_TAILSCALE,
             f"mkdir -p {LAPTOP_OUTPUT_DIR}"],
            check=True, capture_output=True, timeout=10
        )
        result = subprocess.run(
            ["scp", local_path, remote_path],
            capture_output=True, timeout=60
        )
        if result.returncode == 0:
            logger.print(f"[PUSH] Done → D:\\GameDev\\astroforge\\Output\\{run_id}.glb")
            return True
        else:
            logger.print(f"[PUSH] Failed: {result.stderr.decode()}")
            return False
    except Exception as e:
        logger.print(f"[PUSH] Error: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--pipeline-type",     default="512",    choices=["512", "1024"])
    parser.add_argument("--texture-size",      type=int, default=1024)
    parser.add_argument("--decimation-target", type=int, default=100000)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--no-push",           action="store_true", help="Skip SCP to laptop")
    args = parser.parse_args()

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    params = {
        "image":            args.image,
        "pipeline_type":    args.pipeline_type,
        "texture_size":     args.texture_size,
        "decimation_target": args.decimation_target,
        "seed":             args.seed,
    }
    logger   = RunLogger(run_id, params)
    watchdog = Watchdog(WATCHDOG_TIMEOUT, logger)

    try:
        # Step 1 — EnvMap
        watchdog.ping("envmap_load")
        logger.print("[1/7] Loading environment map...")
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread(ENVMAP_PATH, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))
        logger.snapshot("After EnvMap load")

        # Step 2 — Pipeline
        watchdog.ping("pipeline_load")
        logger.print("[2/7] Loading pipeline...")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        if args.pipeline_type == '512':
            # Remove unused 1024 models — saves 5.24 GB VRAM
            del pipeline.models['shape_slat_flow_model_1024']
            del pipeline.models['tex_slat_flow_model_1024']
            gc.collect()
        pipeline.low_vram = False
        pipeline.cuda()
        # BiRefNet.__call__ now auto-casts input to model dtype — no manual cast needed
        logger.snapshot("After pipeline load")

        # Step 3 — Inference (step-by-step to control VRAM between stages)
        image = Image.open(args.image)
        torch.manual_seed(args.seed)

        watchdog.ping("preprocess + cond")
        logger.print("[3a/7] Preprocessing + conditioning...")
        image_proc = pipeline.preprocess_image(image)
        cond = pipeline.get_cond([image_proc], 512)
        logger.snapshot("After get_cond")

        watchdog.ping("sparse structure")
        logger.print("[3b/7] Sampling sparse structure...")
        coords = pipeline.sample_sparse_structure(cond, resolution=32, num_samples=1)
        logger.print(f"  coords: {coords.shape}")
        torch.cuda.empty_cache(); gc.collect()

        watchdog.ping("shape slat")
        logger.print("[3c/7] Sampling shape SLat...")
        shape_slat = pipeline.sample_shape_slat(
            cond, pipeline.models['shape_slat_flow_model_512'], coords
        )
        torch.cuda.empty_cache(); gc.collect()

        watchdog.ping("texture slat")
        logger.print("[3d/7] Sampling texture SLat...")
        tex_slat = pipeline.sample_tex_slat(
            cond, pipeline.models['tex_slat_flow_model_512'], shape_slat
        )
        # Offload flow models to CPU — they're no longer needed after sampling.
        # Flow models: 3 × 2.62 GB = 7.86 GB freed from VRAM.
        # Decoders (shape + tex + sparse): 2.05 GB stay on GPU.
        for name in ['sparse_structure_flow_model', 'shape_slat_flow_model_512', 'tex_slat_flow_model_512']:
            pipeline.models[name].cpu()
        del cond, coords
        torch.cuda.empty_cache(); gc.collect()
        logger.snapshot("After sampling + flow models offloaded")

        watchdog.ping("decode")
        logger.print("[3e/7] Decoding latent to mesh...")
        meshes = pipeline.decode_latent(shape_slat, tex_slat, resolution=512)
        mesh = meshes[0]
        del shape_slat, tex_slat
        torch.cuda.empty_cache(); gc.collect()
        logger.print(f"  Vertices: {mesh.vertices.shape}, Faces: {mesh.faces.shape}")
        logger.snapshot("After decode")

        # Step 4 — Simplify mesh
        watchdog.ping("mesh_simplify")
        logger.print("[4/7] Simplifying mesh...")
        mesh.simplify(16777216)
        logger.print(f"  Simplified → Vertices: {mesh.vertices.shape}, Faces: {mesh.faces.shape}")

        # Step 5 — Free VRAM before GLB export
        watchdog.ping("vram_free")
        logger.print("[5/7] Freeing VRAM for GLB export...")
        for model in pipeline.models.values():
            model.cpu()
        if hasattr(pipeline, 'image_cond_model'):
            pipeline.image_cond_model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        logger.snapshot("After VRAM free (models to CPU)")

        # Step 6 — GLB export
        watchdog.ping("glb_export")
        logger.print("[6/7] Exporting GLB...")
        output_path = str(logger.run_dir / "output.glb")
        glb = o_voxel.postprocess.to_glb(
            vertices            = mesh.vertices,
            faces               = mesh.faces,
            attr_volume         = mesh.attrs,
            coords              = mesh.coords,
            attr_layout         = mesh.layout,
            voxel_size          = mesh.voxel_size,
            aabb                = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   = args.decimation_target,
            texture_size        = args.texture_size,
            remesh              = True,
            remesh_band         = 1,
            remesh_project      = 0,
            verbose             = True,
        )
        glb.export(output_path, extension_webp=True)
        shutil.copy(output_path, "sample.glb")
        logger.snapshot("After GLB export")
        logger.print(f"  Saved: {output_path}")

        # Step 6b — Render preview
        watchdog.ping("render_preview")
        logger.print("[6b/7] Rendering preview...")
        try:
            from glb_render import render_preview
            preview_path = str(logger.run_dir / "preview.png")
            render_preview(output_path, preview_path)
        except Exception as e:
            logger.print(f"  [RENDER] Failed (non-fatal): {e}")
            preview_path = None

        # Step 7 — Push to laptop
        if not args.no_push:
            watchdog.ping("scp_push")
            logger.print("[7/7] Pushing to laptop...")
            push_to_laptop(output_path, run_id, logger)
            if preview_path:
                subprocess.run(
                    ["scp", preview_path, f"{LAPTOP_TAILSCALE}:{LAPTOP_OUTPUT_DIR}/{run_id}_preview.png"],
                    capture_output=True, timeout=30
                )
                logger.print(f"[PUSH] Preview → D:\\GameDev\\astroforge\\Output\\{run_id}_preview.png")
        else:
            logger.print("[7/7] Push skipped (--no-push)")

        watchdog.stop()
        logger.finish(success=True, output_path=output_path)

    except Exception as e:
        watchdog.stop()
        import traceback
        logger.print(traceback.format_exc())
        logger.finish(success=False, error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
