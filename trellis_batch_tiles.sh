#!/bin/bash
# trellis_batch_tiles.sh — liest PNGs aus astroforge/assets/GenAI/png/tiles/
# schreibt GLBs nach astroforge/assets/GenAI/glb/tiles/
source /root/projects/setup_rocm_env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

TRELLIS_DIR="/root/projects/astroforge_trellis_2"
ASTROFORGE="/root/projects/astroforge"
IMG_DIR="$ASTROFORGE/assets/GenAI/png/tiles"
OUT_DIR="$ASTROFORGE/assets/GenAI/glb/tiles"
ONEDRIVE_GLB="/mnt/f/OneDrive/02_AstoForge/glb/tiles"

mkdir -p "$OUT_DIR" "$ONEDRIVE_GLB"
cd "$TRELLIS_DIR"

TILES=(
    "tile_plains"
    "tile_plains_temperate"
    "tile_forest"
    "tile_mountain"
    "tile_water"
    "tile_water_clean"
    "tile_desert"
    "tile_rocky"
    "tile_river_straight"
    "tile_river_bend"
    "tile_resource_iron"
    "tile_resource_coal"
    "tile_resource_copper"
    "tile_resource_gold"
    "tile_resource_oil"
    "tile_resource_silicon"
)

for NAME in "${TILES[@]}"; do
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
    cp "$GLB" "$ONEDRIVE_GLB/${NAME}.glb" 2>/dev/null && \
        echo "  SYNC   → OneDrive"
    echo "  DONE   $NAME → $OUT_DIR/${NAME}.glb  $(date '+%H:%M:%S')"
done

echo ""
echo "════════ ALL TILES DONE $(date '+%H:%M:%S') ════════"
