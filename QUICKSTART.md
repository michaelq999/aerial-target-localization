# C-UASC Green X: macOS Beginner Quickstart

This guide starts from a Mac with **no Blender installed**. The source image is bundled at `reference/competition_target_excerpt.png`.

## Step 1 - Install Blender on your Mac

1. Apple menu > **About This Mac** > check **Chip** or **Processor**.
2. Download Blender from https://www.blender.org/download/ :
   - **Apple Silicon (M-series):** macOS Apple Silicon release (current official stable release).
   - **Intel:** install **Blender 4.5 LTS**, the last Intel-compatible major version: https://www.blender.org/download/lts/4-5/ . Blender 5.x requires Apple Silicon.
3. Open the `.dmg`, drag `Blender.app` to **Applications**, and open it. Allow the first-launch macOS warning if prompted.

## Step 2 - Unzip and inspect project files

1. Double-click the project ZIP and open the extracted `cuasc_green_x/` folder.
2. Open `model/`: `green_x.obj` is the mesh; `green_x.mtl` holds the colors; `spec.json` documents dimensions. **Keep `.obj` and `.mtl` in the same folder.**
3. Compare with `reference/competition_target_excerpt.png`: white square, thin green corner-to-corner diagonals, center intersection. The drawing's thin black border is illustrative; its appearance and line widths are not dimensioned.

## Step 3 - Import the OBJ into Blender

1. Launch Blender, choose **General**, and delete the startup cube (select cube > `X` > Delete).
2. Select **File > Import > Wavefront (.obj)**.
3. Navigate to `cuasc_green_x/model/` and select `green_x.obj`.
4. In import options, set **Forward Axis: Y**, **Up Axis: Z**, and **Scale: 1.0**. The OBJ is written using Blender's Z-up coordinate system.
5. Choose **Import OBJ**.
6. Select **View > Viewpoint > Top**, then **View > Frame All**. Turn on **Material Preview** (shaded sphere icon, viewport upper right), otherwise green and white may appear gray.
7. To verify size, select `White_Target_Board`, press `N` and open Item > Dimensions; the default square is approximately **3.536 m per side**, with a **5 m diagonal**. This size interpretation is an assumption, since the organizer specifies only approximately 5 m diameter.
8. **File > Save As** and save `model/green_x.blend` (or your own working copy).

## Step 4 - Set up VS Code + Python

Open the extracted `cuasc_green_x` folder in VS Code via **File > Open Folder**, then **Terminal > New Terminal**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For PyTorch, Apple Silicon can use MPS acceleration when available; Intel Macs may fall back to CPU. The training Python environment and Blender's bundled Python are **separate**.

## Step 5 - Render aerial training images using Blender

Run in the root project folder in the VS Code terminal (not inside Blender's Python console):

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --python blender/generate_green_x.py -- --out output --samples 10 --size 256 --span-m 5.0 --size-mode diagonal --stroke-width-m 0.06
```

Check the renders in `output/dataset/images/`, automatic center annotations in `output/dataset/labels.csv`, and the generated Blender scene `output/model/green_x.blend`. Then delete the `output/` folder and repeat with `--samples 250` to produce a clean training set.

If you put Blender somewhere other than `/Applications/Blender.app`, modify the path above. Confirm the binary path with:

```bash
ls -l /Applications/Blender.app/Contents/MacOS/Blender
```

## Step 6 - Check labels and run a classical detector

```bash
python vision/check_dataset.py output/dataset
python vision/baseline.py output/dataset/images/frame_00000.png
```

See `output/baseline_result.png` and `output/baseline_result_mask.png`.

## Step 7 - Train and predict with the heatmap model

```bash
python vision/train.py --dataset output/dataset --epochs 20 --batch-size 8
IMG=$(head -n 1 output/validation_files.txt)
python vision/predict.py "output/dataset/images/$IMG"
```

See `output/prediction_overlay.png` and training metrics. Success is `(x, y)` **pixel center** plus validation error, not yet a geolocation or a calibrated object-present probability.

## Target size limitations

The official drawing and text show a white square with a green X, approximately **5 m diameter**, but **no exact stripe width or square-side dimension**. We set a 5 m **diagonal** and 0.06 m **stripe width** as editable approximations, emphasizing relatively thin strokes matching the reference's appearance. In a future round, check with organizers whether "5 m diameter" means diagonal or side length. To model a 5 m side instead, use `--size-mode side --span-m 5.0`.
