"""
glb_render.py — headless GLB renderer (EGL, no display needed)
Renders 4 views (isometric, front, side, top) + mesh stats into a PNG preview grid.

Usage:
    python glb_render.py path/to/model.glb [output.png]
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
import math
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

RESOLUTION = 512
BG_COLOR   = (24, 24, 32, 255)
LABEL_BG   = (0, 0, 0, 180)


# ── Camera helpers ────────────────────────────────────────────────────────────

def _look_at(eye, target, up=np.array([0, 1, 0])):
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        up = np.array([0, 0, 1])
        x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0] = x
    m[:3, 1] = y
    m[:3, 2] = z
    m[:3, 3] = eye
    return m


def _render_view(scene_tri, eye_offset, bounds, res=RESOLUTION):
    center = (bounds[0] + bounds[1]) / 2
    extents = bounds[1] - bounds[0]
    radius  = np.linalg.norm(extents) * 0.8

    eye = center + np.array(eye_offset) * radius
    cam_pose = _look_at(eye, center)

    ps = pyrender.Scene(bg_color=BG_COLOR, ambient_light=[0.3, 0.3, 0.3])

    # Add meshes
    for geom in scene_tri.geometry.values():
        if isinstance(geom, trimesh.Trimesh):
            try:
                # Let pyrender resolve textures automatically from trimesh material
                mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
            except Exception:
                mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
            ps.add(mesh)

    # Lights
    dl = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    dl2 = pyrender.DirectionalLight(color=np.ones(3) * 0.8, intensity=1.5)
    ps.add(dl,  pose=_look_at(center + np.array([1, 2, 1]) * radius, center))
    ps.add(dl2, pose=_look_at(center + np.array([-1, 1, -1]) * radius, center))

    # Camera — use FOV to frame the object
    fov = math.atan2(np.linalg.norm(extents) * 0.65, np.linalg.norm(eye - center)) * 2
    cam = pyrender.PerspectiveCamera(yfov=max(fov, 0.3), aspectRatio=1.0)
    ps.add(cam, pose=cam_pose)

    r = pyrender.OffscreenRenderer(res, res)
    try:
        color, _ = r.render(ps)
    finally:
        r.delete()
    return Image.fromarray(color)


def _add_label(img, text):
    draw = ImageDraw.Draw(img, 'RGBA')
    draw.rectangle([0, img.height - 22, img.width, img.height], fill=LABEL_BG)
    draw.text((6, img.height - 18), text, fill=(200, 200, 200))
    return img


# ── Stats panel ───────────────────────────────────────────────────────────────

def _stats_panel(scene_tri, glb_path, res=RESOLUTION):
    img = Image.new('RGBA', (res, res), BG_COLOR)
    draw = ImageDraw.Draw(img)

    verts = sum(len(g.vertices) for g in scene_tri.geometry.values() if isinstance(g, trimesh.Trimesh))
    faces = sum(len(g.faces)    for g in scene_tri.geometry.values() if isinstance(g, trimesh.Trimesh))
    mats  = len(scene_tri.geometry)
    size_mb = os.path.getsize(glb_path) / 1e6

    bounds = scene_tri.bounds
    extents = bounds[1] - bounds[0] if bounds is not None else np.zeros(3)

    lines = [
        "--- Mesh Stats ---",
        f"  Vertices  : {verts:,}",
        f"  Faces     : {faces:,}",
        f"  Geometries: {mats}",
        f"  File size : {size_mb:.2f} MB",
        "",
        "--- Bounding Box ---",
        f"  X : {extents[0]:.3f}",
        f"  Y : {extents[1]:.3f}",
        f"  Z : {extents[2]:.3f}",
        "",
        "--- Textures ---",
    ]

    # Texture info
    tex_found = False
    for name, geom in scene_tri.geometry.items():
        if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
            mat = geom.visual.material
            tex = None
            for attr in ('baseColorTexture', 'image'):
                candidate = getattr(mat, attr, None)
                if candidate is None:
                    continue
                # baseColorTexture wraps an image in a Texture object
                tex = getattr(candidate, 'image', candidate)
                if hasattr(tex, 'size'):
                    break
                tex = None
            if tex is not None and hasattr(tex, 'size'):
                lines.append(f"  {name[:20]}: {tex.size[0]}×{tex.size[1]}")
                tex_found = True
    if not tex_found:
        lines.append("  (no texture info)")

    y = 20
    for line in lines:
        draw.text((16, y), line, fill=(180, 210, 180))
        y += 22

    return img.convert('RGB')


# ── Main ──────────────────────────────────────────────────────────────────────

def render_preview(glb_path: str, output_path: str = None) -> str:
    if output_path is None:
        output_path = glb_path.replace('.glb', '_preview.png')

    print(f"[RENDER] Loading {glb_path}...", flush=True)
    scene = trimesh.load(glb_path, force='scene')

    if scene.bounds is None or len(scene.geometry) == 0:
        print("[RENDER] Empty scene, skipping.", flush=True)
        return None

    bounds = scene.bounds

    # 3 views + stats panel
    views = [
        ([1.4, 1.0, 1.4],   "Isometric"),
        ([0.0, 0.1, 2.0],   "Front"),
        ([2.0, 0.1, 0.0],   "Right"),
    ]

    renders = []
    for eye, label in views:
        print(f"[RENDER] View: {label}...", flush=True)
        try:
            img = _render_view(scene, eye, bounds)
            img = img.convert('RGBA')
            _add_label(img, label)
            renders.append(img.convert('RGB'))
        except Exception as e:
            print(f"[RENDER] View {label} failed: {e}", flush=True)
            renders.append(Image.new('RGB', (RESOLUTION, RESOLUTION), (40, 40, 40)))

    stats = _stats_panel(scene, glb_path)

    # 2×2 grid: isometric | front / right | stats
    grid = Image.new('RGB', (RESOLUTION * 2, RESOLUTION * 2), (10, 10, 16))
    positions = [(0, 0), (RESOLUTION, 0), (0, RESOLUTION), (RESOLUTION, RESOLUTION)]
    panels = [renders[0], renders[1], renders[2], stats]
    for img, pos in zip(panels, positions):
        grid.paste(img, pos)

    grid.save(output_path)
    print(f"[RENDER] Saved preview → {output_path}", flush=True)
    return output_path


if __name__ == '__main__':
    glb = sys.argv[1] if len(sys.argv) > 1 else 'sample.glb'
    out = sys.argv[2] if len(sys.argv) > 2 else None
    render_preview(glb, out)
