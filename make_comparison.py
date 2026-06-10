"""
make_comparison.py — side-by-side Meshy vs Trellis 2 comparison grid
"""
from PIL import Image, ImageDraw
import sys

def compare(meshy_path, trellis_path, output_path, label_left="Meshy", label_right="Trellis 2"):
    m = Image.open(meshy_path).convert("RGB")
    t = Image.open(trellis_path).convert("RGB")
    W, H = m.width, m.height
    out = Image.new("RGB", (W * 2 + 8, H + 32), (10, 10, 16))
    out.paste(m, (0, 32))
    out.paste(t, (W + 8, 32))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, W, 30], fill=(30, 60, 30))
    draw.rectangle([W + 8, 0, W * 2 + 8, 30], fill=(30, 30, 80))
    draw.text((8, 6), label_left, fill=(180, 255, 180))
    draw.text((W + 16, 6), label_right, fill=(180, 180, 255))
    out.save(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    # factory
    compare(
        "/root/projects/astroforge/assets/Trellis/GLB/building_factory_preview.png",
        "/root/projects/astroforge_trellis_2/runs/run_20260526_162340/preview.png",
        "/root/projects/astroforge_trellis_2/compare_factory.png",
        "Meshy — building_factory",
        "Trellis 2 — run_162340"
    )
    # command center
    compare(
        "/root/projects/astroforge/assets/Trellis/GLB/building_command_center_preview.png",
        "/root/projects/astroforge_trellis_2/runs/run_20260526_161016/preview.png",
        "/root/projects/astroforge_trellis_2/compare_command_center.png",
        "Meshy — building_command_center",
        "Trellis 2 — run_161016"
    )
    # sample (first trellis2 run)
    compare(
        "/root/projects/astroforge/assets/Trellis/GLB/sample_2026-05-24T203333.099_preview.png",
        "/root/projects/astroforge_trellis_2/runs/run_20260526_152537/preview.png",
        "/root/projects/astroforge_trellis_2/compare_sample.png",
        "Meshy — sample_2026-05-24",
        "Trellis 2 — run_152537"
    )
