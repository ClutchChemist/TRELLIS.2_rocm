import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
os.environ["HSA_XNACK"] = "1"

import cv2
import time
import psutil
import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
import o_voxel


# ── Resource snapshot ─────────────────────────────────────────────────────────

def snapshot(label: str, t0: float):
    vm   = psutil.virtual_memory()
    sw   = psutil.swap_memory()
    cpu  = psutil.cpu_percent(interval=0.3)
    vram_alloc = torch.cuda.memory_allocated() / 1e9
    vram_res   = torch.cuda.memory_reserved()  / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    elapsed    = time.time() - t0
    print(
        f"\n{'='*62}\n"
        f"  {label}\n"
        f"{'='*62}\n"
        f"  Elapsed   : {elapsed:7.1f}s\n"
        f"  CPU       : {cpu:5.1f}%\n"
        f"  RAM       : {vm.used/1e9:5.2f} / {vm.total/1e9:.2f} GB  ({vm.percent:.1f}%)\n"
        f"  Swap      : {sw.used/1e9:5.2f} / {sw.total/1e9:.2f} GB  ({sw.percent:.1f}%)\n"
        f"  VRAM alloc: {vram_alloc:5.2f} GB\n"
        f"  VRAM reserv:{vram_res:5.2f} / {vram_total:.2f} GB\n",
        flush=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────

T0 = time.time()
snapshot("START", T0)

# Step 1 — EnvMap
print("[STEP 1] Loading environment map...", flush=True)
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))
snapshot("After EnvMap load", T0)

# Step 2 — Pipeline
print("[STEP 2] Loading pipeline...", flush=True)
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
import gc
del pipeline.models['shape_slat_flow_model_1024']
del pipeline.models['tex_slat_flow_model_1024']
gc.collect()
pipeline.low_vram = False  # 9.9 GB models fit in 15.8 GB VRAM after removing unused 1024 models
pipeline.cuda()
snapshot("After pipeline load (1024 models removed, low_vram=False)", T0)

# Step 3 — Preprocess image + conditioning
print("[STEP 3] Preprocessing image...", flush=True)
image = Image.open("assets/example_image/T.png")
torch.manual_seed(42)
image_proc = pipeline.preprocess_image(image)
snapshot("After image preprocess", T0)

print("[STEP 3] Computing conditioning...", flush=True)
cond = pipeline.get_cond([image_proc], 512)
snapshot("After get_cond (DINO/CLIP on GPU)", T0)

# Step 4 — Sparse structure sampling
print("[STEP 4] Sampling sparse structure...", flush=True)
coords = pipeline.sample_sparse_structure(cond, resolution=32, num_samples=1)
print(f"  coords shape: {coords.shape}", flush=True)
snapshot("After sparse structure sampling", T0)

# Step 5 — Shape SLat sampling
print("[STEP 5] Sampling shape SLat...", flush=True)
shape_slat = pipeline.sample_shape_slat(
    cond, pipeline.models['shape_slat_flow_model_512'], coords
)
snapshot("After shape SLat sampling", T0)

# Step 6 — Texture SLat sampling
print("[STEP 6] Sampling texture SLat...", flush=True)
tex_slat = pipeline.sample_tex_slat(
    cond, pipeline.models['tex_slat_flow_model_512'], shape_slat
)
snapshot("After texture SLat sampling", T0)

# Step 7 — Decode latent → mesh
print("[STEP 7] Decoding latent to mesh...", flush=True)
meshes = pipeline.decode_latent(shape_slat, tex_slat, resolution=512)
mesh = meshes[0]
print(f"  Vertices: {mesh.vertices.shape}, Faces: {mesh.faces.shape}", flush=True)
snapshot("After decode (mesh ready)", T0)

# Step 8 — Simplify
print("[STEP 8] Simplifying mesh...", flush=True)
mesh.simplify(16777216)
print(f"  Simplified → Vertices: {mesh.vertices.shape}, Faces: {mesh.faces.shape}", flush=True)
snapshot("After mesh simplify", T0)

# Step 9 — GLB export
print("[STEP 8b] Freeing VRAM before GLB export...", flush=True)
for model in pipeline.models.values():
    model.cpu()
if hasattr(pipeline, 'image_cond_model'):
    pipeline.image_cond_model.cpu()
torch.cuda.empty_cache()
gc.collect()
snapshot("After VRAM free (models to CPU)", T0)

print("[STEP 9] Exporting GLB...", flush=True)
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   100000,
    texture_size        =   1024,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export("sample_profiled.glb", extension_webp=True)
snapshot("DONE — GLB exported", T0)
print("[STEP 9] Done! sample_profiled.glb exported.", flush=True)
