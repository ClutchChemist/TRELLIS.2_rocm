# Trellis 2 — ROCm/AMD Setup & Betriebsdokumentation

**System:** AMD Ryzen 7800X3D + Radeon RX 7800 XT (16 GB VRAM)  
**OS:** WSL2 (Ubuntu) auf Windows 11  
**Stack:** ROCm 7.2.1, PyTorch ROCm, Python 3.11  
**Modell:** `microsoft/TRELLIS.2-4B` (14 GB, HuggingFace Cache)  
**Projekt:** `/root/projects/astroforge_trellis_2/`  
**Stand:** 2026-05-26 — vollständig funktionierend

---

## Trellis 1 vs Trellis 2

| | Trellis 1 | Trellis 2 |
|---|---|---|
| Modell | `TRELLIS-image-large` (2.9 GB) | `TRELLIS.2-4B` (14 GB) |
| Projektordner | `/root/projects/astroforge_trellis/` | `/root/projects/astroforge_trellis_2/` |
| Qualität | geringer | deutlich besser (4B Parameter) |
| Status | **nicht mehr genutzt** | **aktiv** |

Trellis 1 kann gelöscht werden (spart ~3.5 GB):
```bash
rm -rf /root/projects/astroforge_trellis/
rm -rf /root/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/
```

---

## Schnellstart

```bash
source /root/projects/setup_rocm_env.sh
source /root/projects/astroforge_trellis_2/.venv/bin/activate
cd /root/projects/astroforge_trellis_2

python run_trellis.py assets/example_image/T.png
# → runs/run_YYYYMMDD_HHMMSS/output.glb + automatisch auf Laptop per SCP
```

---

## Produktiver Workflow — `run_trellis.py`

Der einzige Einstiegspunkt für Produktion. Ersetzt alle `test_gpu*.py`-Skripte.

```bash
# Standard
python run_trellis.py bild.png

# Mit eigenen Parametern
python run_trellis.py bild.png --texture-size 2048 --decimation-target 200000 --seed 123

# Ohne Push auf Laptop
python run_trellis.py bild.png --no-push
```

**Jeder Run erzeugt `runs/run_YYYYMMDD_HHMMSS/`:**
- `output.glb` — fertiges 3D-Modell (WebP-Texturen)
- `run.log` — Plaintext mit Ressourcen-Snapshots pro Step
- `run.json` — strukturiertes JSON (alle Metriken, Parameter, Ergebnis)

**Features:**
- Watchdog-Thread bricht automatisch ab wenn Step > 300s hängt
- Auto-SCP via Tailscale nach Erfolg → `D:\GameDev\astroforge\Output\<run_id>.glb`
- Step-by-Step Inference mit explizitem VRAM-Management (kein OOM)

---

## VRAM-Strategie (RX 7800 XT, 15.8 GB effektiv)

Das 4B-Modell hat 15.14 GB Gesamtgewichte — passt nicht mit `low_vram=False` naiv rein.

**Lösung in `run_trellis.py`:**

1. **1024er-Modelle gar nicht laden** — für `pipeline_type='512'` nicht nötig, spart 5.24 GB
2. **Flow-Modelle nach Sampling auf CPU** — nach Sparse/Shape/Tex-SLat Sampling werden die Flow-Modelle (7.86 GB) auf CPU geschoben, Decoder (2.05 GB) bleiben auf GPU
3. **`low_vram=False` während Inference** — alle aktiven Modelle permanent auf GPU, keine CPU↔GPU-Transfers

```
Modell                          Größe    Wann auf GPU
───────────────────────────────────────────────────────
sparse_structure_flow_model     2.62 GB  Sparse sampling → dann CPU
shape_slat_flow_model_512       2.62 GB  Shape SLat → dann CPU
tex_slat_flow_model_512         2.62 GB  Tex SLat → dann CPU
sparse_structure_decoder        0.15 GB  Decode (bleibt)
shape_slat_decoder              0.95 GB  Decode (bleibt)
tex_slat_decoder                0.95 GB  Decode (bleibt)
shape_slat_flow_model_1024      2.62 GB  NICHT geladen (nur '1024' pipeline)
tex_slat_flow_model_1024        2.62 GB  NICHT geladen (nur '1024' pipeline)
```

**VRAM-Verlauf pro Phase:**

| Phase | VRAM | RAM |
|-------|------|-----|
| Pipeline geladen (ohne 1024er) | 11.70 GB | 15.7 GB |
| Sampling (alle 3 Flow-Modelle) | 11.75 GB | 15.9 GB |
| Flow-Modelle → CPU | 0.62 GB | 16.4 GB |
| Decode + fill_holes | ~3–4 GB | 16.4 GB |
| GLB Export (cumesh+xatlas) | 3.15 GB | 16.5 GB |

**Gesamtlaufzeit:** ~241s bei `texture_size=1024`, `decimation_target=100000`

---

## Gelöste Probleme

### Problem 1: `ModuleNotFoundError: No module named 'flash_attn'`

**Wann:** Beim zweiten Sampling-Step (`sample_shape_slat`).  
**Ursache:** `flash_attn` ist NVIDIA-only. Sparse-Attention-Config hatte `sdpa` nicht in der Whitelist — trotz `ATTN_BACKEND=sdpa` in der Umgebung wurde der Default `flash_attn` benutzt.

**Fix — 3 Dateien:**

`trellis2/modules/sparse/config.py`:
```python
# VORHER: ATTN = 'flash_attn'  +  whitelist ohne 'sdpa'
ATTN = 'sdpa'
if env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']:
```

`trellis2/modules/sparse/attention/full_attn.py` — sdpa-Branch mit chunk-loop über variable Sequenzlängen:
```python
if config.ATTN == 'sdpa':
    import torch.nn.functional as F
    # chunk-loop über q_seqlen / kv_seqlen ...
elif config.ATTN == 'xformers':
    ...
```

`trellis2/modules/sparse/attention/windowed_attn.py` — sdpa-Branch in allen drei Funktionen:
- `calc_window_partition`: `else: attn_func_args = {'seq_lens': seq_lens}`
- `sparse_windowed_scaled_dot_product_self_attention`: sdpa chunk-loop
- `sparse_windowed_scaled_dot_product_cross_attention`: sdpa chunk-loop

---

### Problem 2: Pipeline hängt bei `texture_size=4096`

**Wann:** GLB-Export, "Sampling attributes..." — hängt 30+ Minuten.  
**Ursache:** nvdiffrast `drawTriangles` auf ROCm/DXG bei 4096×4096 = 16 MP. Deadlock oder extremes Performance-Problem mit DirectX GPU-Passthrough bei dieser Auflösung.  
**Fix:** `texture_size=1024` (läuft stabil). `texture_size=2048` nicht getestet.

---

### Problem 3: WSL2 RAM-Limit

**Ursache:** WSL2-Default ~8 GB RAM → OOM während Pipeline-Inferenz.  
**Fix:** `C:\Users\Andrej König\.wslconfig`:
```ini
[wsl2]
memory=22GB
swap=40GB
swapfile=E:\wsl-swap.vhdx
```
Nach `wsl --shutdown` + Neustart: 22 GB RAM + 40 GB Swap.

---

### Problem 4: OOM in `fill_holes` / `get_edges` (cumesh)

**Wann:** Decode-Step, auch nach Entfernen der 1024er-Modelle.  
**Ursache:** Flow-Modelle (7.86 GB) belegten noch VRAM während cumesh `get_edges()` auf dem ~1.4M-Vertex-Mesh arbeitete.  
**Fix:** Flow-Modelle nach dem Sampling explizit auf CPU schieben, `torch.cuda.empty_cache()` aufrufen — dann hat cumesh ~12 GB Puffer.

---

### Problem 5: `RuntimeError: Input type (float) and bias type (c10::Half)` bei JPEG/RGB-Input

**Wann:** `preprocess_image()` → `BiRefNet.__call__()` — tritt bei Bildern MIT Hintergrund auf (JPEG, RGB-PNG). Transparente Bilder (RGBA) überspringen Background-Removal und sind nicht betroffen.  
**Ursache:** `BiRefNet` (Wrapper um HuggingFace `AutoModelForImageSegmentation`) lädt Gewichte in fp16, aber `transforms.ToTensor()` erzeugt fp32-Tensoren. Mismatch beim ersten Conv-Layer.  
**Fix in `trellis2/pipelines/rembg/BiRefNet.py`:**
```python
# __call__: input dtype an Modell anpassen statt hardcoded float32
dtype = next(self.model.parameters()).dtype
input_images = self.transform_image(image).unsqueeze(0).to("cuda", dtype=dtype)

# float() Methode ergänzt damit externe .float()-Aufrufe funktionieren
def float(self):
    self.model.float()
```

---

### Problem 6: `low_vram=True` — RAM voll, VRAM leer

**Symptom:** RAM bei 99.4% (22.9/23 GB), VRAM allokiert nur 0.05 GB.  
**Ursache:** `low_vram=True` schiebt alle Gewichte (~15 GB) in CPU-RAM. GPU bekommt Modelle nur kurz pro Step.  
**Fix:** Strategisches VRAM-Management (siehe oben) statt `low_vram=True`.

---

## ROCm-Besonderheiten in WSL2

- **Kein `/dev/kfd`** — WSL2 nutzt DirectX GPU-Passthrough via `/dev/dxg`
- **`rocm-smi` funktioniert nicht** in WSL2 — `amdgpu`-Kernelmodul fehlt, GPU-Utilization-Monitoring nicht möglich
- **GPU-Maskierung:** RX 7800 XT ist gfx1102, läuft als gfx1100 (`HSA_OVERRIDE_GFX_VERSION=11.0.0`)
- **Attention-Backend:** `sdpa` (PyTorch-nativ, `torch.nn.functional.scaled_dot_product_attention`) — kein `flash_attn`, kein `xformers`

**Umgebungsvariablen** (gesetzt durch `/root/projects/setup_rocm_env.sh`):
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_ENABLE_DXG_DETECTION=1
export ATTN_BACKEND=sdpa
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export HSA_XNACK=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

## Tailscale — Desktop ↔ Laptop

| Gerät | Hostname | Tailscale-IP |
|-------|----------|--------------|
| Desktop (WSL) | maet-desktop | `100.96.166.68` |
| Laptop (WSL) | andrej-laptopyoga | `100.74.177.32` |

SCP Desktop → Laptop:
```bash
scp datei.glb root@100.74.177.32:/mnt/d/GameDev/astroforge/Output/
```

`run_trellis.py` macht das automatisch nach jedem erfolgreichen Run.  
**GLB ansehen:** Windows 3D-Viewer (Doppelklick) oder Blender → File → Import → glTF 2.0

---

## Datei-Übersicht

| Datei | Zweck |
|-------|-------|
| `run_trellis.py` | **Produktiver Einstiegspunkt** — CLI, Logging, Watchdog, Auto-Push |
| `test_gpu.py` | Einfacher Testlauf ohne Logging |
| `test_gpu_profiled.py` | Detailliertes Profiling-Skript (Step-by-Step Snapshots) |
| `runs/` | Alle Run-Outputs (GLB, log, JSON) |
| `assets/example_image/T.png` | Referenz-Testbild |
| `assets/hdri/forest.exr` | EnvMap für Beleuchtung |
| `/root/projects/setup_rocm_env.sh` | ROCm-Umgebungsvariablen |
