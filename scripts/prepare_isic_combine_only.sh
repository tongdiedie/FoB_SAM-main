#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SRC_IMG_ROOT="data/isic/images"
SRC_GT_ROOT="data/isic/ISIC2018_Task1_Training_GroundTruth"

OUT_ROOT="data/isic/combine"
OUT_IMG_ROOT="${OUT_ROOT}/ISIC2018_Task1-2_Training_Input"
OUT_GT_ROOT="${OUT_ROOT}/ISIC2018_Task1_Training_GroundTruth"

echo "============================================================"
echo "Prepare ISIC Setting II combine only"
echo "This script will NOT delete ISIC_setting_1, raw images, masks, or superpixels."
echo "============================================================"

if [ ! -d "$SRC_IMG_ROOT/melanoma" ]; then
  echo "ERROR: missing $SRC_IMG_ROOT/melanoma"
  echo "You probably need to run: python data/isic/split.py"
  exit 1
fi

if [ ! -d "$SRC_IMG_ROOT/nevus" ]; then
  echo "ERROR: missing $SRC_IMG_ROOT/nevus"
  echo "You probably need to run: python data/isic/split.py"
  exit 1
fi

if [ ! -d "$SRC_IMG_ROOT/seborrheic_keratosis" ]; then
  echo "ERROR: missing $SRC_IMG_ROOT/seborrheic_keratosis"
  echo "You probably need to run: python data/isic/split.py"
  exit 1
fi

if [ ! -d "$SRC_GT_ROOT" ]; then
  echo "ERROR: missing $SRC_GT_ROOT"
  exit 1
fi

# 只在你显式设置 REBUILD_COMBINE=1 时删除 combine
if [ "${REBUILD_COMBINE:-0}" = "1" ]; then
  echo "REBUILD_COMBINE=1 detected, removing old combine only:"
  echo "$OUT_ROOT"
  rm -rf "$OUT_ROOT"
fi

mkdir -p "$OUT_IMG_ROOT/1"
mkdir -p "$OUT_IMG_ROOT/2"
mkdir -p "$OUT_IMG_ROOT/3"
mkdir -p "$OUT_GT_ROOT"

echo "[1/4] Copy melanoma -> class 1"
find "$SRC_IMG_ROOT/melanoma" -maxdepth 1 -type f -name "*.jpg" -exec cp -n {} "$OUT_IMG_ROOT/1/" \;

echo "[2/4] Copy nevus -> class 2"
find "$SRC_IMG_ROOT/nevus" -maxdepth 1 -type f -name "*.jpg" -exec cp -n {} "$OUT_IMG_ROOT/2/" \;

echo "[3/4] Copy seborrheic_keratosis -> class 3"
find "$SRC_IMG_ROOT/seborrheic_keratosis" -maxdepth 1 -type f -name "*.jpg" -exec cp -n {} "$OUT_IMG_ROOT/3/" \;

echo "[4/4] Copy ground truth masks"
find "$SRC_GT_ROOT" -maxdepth 1 -type f -name "*_segmentation.png" -exec cp -n {} "$OUT_GT_ROOT/" \;

echo "============================================================"
echo "Check combine result"
echo "============================================================"

echo "class 1 melanoma:"
find "$OUT_IMG_ROOT/1" -maxdepth 1 -type f -name "*.jpg" | wc -l

echo "class 2 nevus:"
find "$OUT_IMG_ROOT/2" -maxdepth 1 -type f -name "*.jpg" | wc -l

echo "class 3 seborrheic_keratosis:"
find "$OUT_IMG_ROOT/3" -maxdepth 1 -type f -name "*.jpg" | wc -l

echo "ground truth masks:"
find "$OUT_GT_ROOT" -maxdepth 1 -type f -name "*_segmentation.png" | wc -l

echo "Done. combine is ready at:"
echo "$OUT_ROOT"
