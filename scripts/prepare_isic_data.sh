#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RAW_IMG="data/isic/ISIC2018_Task1-2_Training_Input"
RAW_GT="data/isic/ISIC2018_Task1_Training_GroundTruth"
S1_DIR="data/isic/ISIC_setting_1"
S2_DIR="data/isic/combine"

echo "============================================================"
echo "Preparing ISIC data"
echo "============================================================"

if [ ! -d "$RAW_IMG" ]; then
  echo "ERROR: missing $RAW_IMG"
  exit 1
fi

if [ ! -d "$RAW_GT" ]; then
  echo "ERROR: missing $RAW_GT"
  exit 1
fi

if [ ! -f "data/isic/class_id.csv" ]; then
  echo "ERROR: missing data/isic/class_id.csv"
  exit 1
fi

echo "[1/4] Split images by lesion class"
rm -rf data/isic/images data/isic/masks
python data/isic/split.py

echo "[2/4] Prepare Setting I folders"
mkdir -p "$S1_DIR/images" "$S1_DIR/gt"

find "$S1_DIR/images" -maxdepth 1 -type f -name "*.jpg" -delete || true
find "$S1_DIR/gt" -maxdepth 1 -type f -name "*.png" -delete || true

cp "$RAW_IMG"/*.jpg "$S1_DIR/images/"
cp "$RAW_GT"/*_segmentation.png "$S1_DIR/gt/"

if [ ! -f "$S1_DIR/isic_in_5_folds.json" ]; then
  echo "ERROR: missing $S1_DIR/isic_in_5_folds.json"
  echo "This file should come with the FoB_SAM repository."
  exit 1
fi

echo "[3/4] Generate Setting I superpixels"
rm -rf "$S1_DIR/superpixels"
python data/isic/prepare_setting1_dataset.py

echo "[4/4] Prepare Setting II combine folders"
rm -rf "$S2_DIR"

mkdir -p "$S2_DIR/ISIC2018_Task1-2_Training_Input/1"
mkdir -p "$S2_DIR/ISIC2018_Task1-2_Training_Input/2"
mkdir -p "$S2_DIR/ISIC2018_Task1-2_Training_Input/3"
mkdir -p "$S2_DIR/ISIC2018_Task1_Training_GroundTruth"

# Numeric class mapping used for Setting II:
# 1 = melanoma, 2 = nevus, 3 = seborrheic_keratosis
cp data/isic/images/melanoma/*.jpg \
  "$S2_DIR/ISIC2018_Task1-2_Training_Input/1/"

cp data/isic/images/nevus/*.jpg \
  "$S2_DIR/ISIC2018_Task1-2_Training_Input/2/"

cp data/isic/images/seborrheic_keratosis/*.jpg \
  "$S2_DIR/ISIC2018_Task1-2_Training_Input/3/"

cp "$RAW_GT"/*_segmentation.png \
  "$S2_DIR/ISIC2018_Task1_Training_GroundTruth/"

echo "============================================================"
echo "Checking ISIC data"
echo "============================================================"

echo "Raw images:"
find "$RAW_IMG" -maxdepth 1 -name "*.jpg" | wc -l

echo "Raw masks:"
find "$RAW_GT" -maxdepth 1 -name "*_segmentation.png" | wc -l

echo "Setting I images:"
find "$S1_DIR/images" -maxdepth 1 -name "*.jpg" | wc -l

echo "Setting I gt:"
find "$S1_DIR/gt" -maxdepth 1 -name "*_segmentation.png" | wc -l

echo "Setting I superpixels:"
find "$S1_DIR/superpixels" -maxdepth 1 -name "*_mask.png" | wc -l

echo "Setting II class 1 melanoma:"
find "$S2_DIR/ISIC2018_Task1-2_Training_Input/1" -maxdepth 1 -name "*.jpg" | wc -l

echo "Setting II class 2 nevus:"
find "$S2_DIR/ISIC2018_Task1-2_Training_Input/2" -maxdepth 1 -name "*.jpg" | wc -l

echo "Setting II class 3 seborrheic_keratosis:"
find "$S2_DIR/ISIC2018_Task1-2_Training_Input/3" -maxdepth 1 -name "*.jpg" | wc -l

echo "Setting II gt:"
find "$S2_DIR/ISIC2018_Task1_Training_GroundTruth" -maxdepth 1 -name "*_segmentation.png" | wc -l

echo "ISIC preparation finished."
