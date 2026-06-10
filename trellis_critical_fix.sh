#!/bin/bash
source /root/projects/setup_rocm_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

TRELLIS_DIR="/root/projects/astroforge_trellis_2"
ASTROFORGE="/root/projects/astroforge"
IMG_DIR="$ASTROFORGE/assets/GenAI/png/tiles"
OUT_DIR="$ASTROFORGE/assets/GenAI/glb/tiles"
ONEDRIVE_GLB="/mnt/f/OneDrive/02_AstoForge/glb/tiles"

mkdir -p "$OUT_DIR" "$ONEDRIVE_GLB"
cd "$TRELLIS_DIR"

for NAME in tile_water tile_resource_gold tile_resource_copper; do
    echo ""
    echo "════════════════════════════════════════"
    echo "  START  $NAME  $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════"

    IMG="$IMG_DIR/${NAME}_v2.png"
    [ ! -f "$IMG" ] && echo "  SKIP — $IMG not found" && continue

    .venv/bin/python run_trellis.py "$IMG" --no-push

    LATEST=$(ls -t runs/ | grep '^run_' | head -1)
    GLB="runs/$LATEST/output.glb"
    [ ! -f "$GLB" ] && echo "  FAIL — no output.glb in $LATEST" && continue

    cp "$GLB" "$OUT_DIR/${NAME}.glb"
    cp "$GLB" "$ONEDRIVE_GLB/${NAME}.glb" 2>/dev/null && echo "  SYNC   → OneDrive"
    echo "  DONE   $NAME  $(date '+%H:%M:%S')"
done

echo ""
echo "════════ CRITICAL FIX DONE $(date '+%H:%M:%S') ════════"
