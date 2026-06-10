import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
os.environ["HSA_XNACK"] = "1"

import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 1. Setup Environment Map
print("[STEP 1] Loading environment map...")
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
# pipeline.json in HF cache is already patched to use ZhengPeng7/BiRefNet (public, cached locally)
print("[STEP 2] Loading pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
# Remove 1024-res models — only pipeline_type='512' is used here (saves 5.24 GB VRAM)
import gc
del pipeline.models['shape_slat_flow_model_1024']
del pipeline.models['tex_slat_flow_model_1024']
gc.collect()
pipeline.low_vram = False  # ~9.9 GB models + ~2.8 GB overhead fits in 15.8 GB VRAM
pipeline.cuda()
print("[STEP 2] Pipeline loaded.")

# 3. Load Image & Run
print("[STEP 3] Running image-to-3D inference...")
image = Image.open("assets/example_image/T.png")
print(f"[STEP 3] GPU mem before run: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
mesh = pipeline.run(image, pipeline_type='512')[0]
print(f"[STEP 3] Inference done. Vertices: {mesh.vertices.shape}, Faces: {mesh.faces.shape}")
mesh.simplify(16777216) # nvdiffrast limit
print(f"[STEP 3] Simplified. Vertices: {mesh.vertices.shape}, Faces: {mesh.faces.shape}")

# 4. Free VRAM before GLB export (cumesh simplifier needs GPU memory)
print("[STEP 4] Freeing VRAM before GLB export...")
for model in pipeline.models.values():
    model.cpu()
if hasattr(pipeline, 'image_cond_model'):
    pipeline.image_cond_model.cpu()
torch.cuda.empty_cache()
gc.collect()
print(f"[STEP 4] VRAM after free: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated / {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")

# 5. Export to GLB
print("[STEP 5] Exporting GLB...")
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
glb.export("sample.glb", extension_webp=True)
print("[STEP 5] Done! sample.glb exported successfully.")
