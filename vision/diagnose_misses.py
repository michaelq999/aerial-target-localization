from baseline_v2 import verify_green_x
import cv2
from pathlib import Path
from baseline_v2 import detect

dataset = Path("output_test/dataset/images")

misses = [
    19, 35, 68, 69, 77, 137, 141,
    154, 174, 178, 187, 189, 190
]

for i in misses:
    path = dataset / f"frame_{i:05d}.png"
    img = cv2.imread(str(path))

    if img is None:
        print(f"Frame {i:03d}: IMAGE MISSING")
        continue

    center_white, _, corners = detect(
        img, verify_green=False
    )

    if center_white is None:
        print(f"Frame {i:03d}: GEOMETRY FAILURE")
        continue

    print(f"\n--- Frame {i:03d} ---")

    accepted, score, warped, evidence = verify_green_x(
        img, corners, threshold=0.030
    )

    print(
        f"Frame {i:03d}: "
        f"green_score={score:.4f}, "
        f"accepted={accepted}"
    )

    # Save visualization of the rejected target.
    out = Path("output/green_diagnostics")
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / f"frame_{i:05d}_warped.png"), warped)
    cv2.imwrite(str(out / f"frame_{i:05d}_evidence.png"), evidence) 
