"""
Generate side-by-side comparison GIFs (MUSCL vs PINN) for all topologies.
"""
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

from topology_config import get_all_topology_names

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIFS_DIR = os.path.join(BASE_DIR, "gifs")
os.makedirs(GIFS_DIR, exist_ok=True)

for topo_name in get_all_topology_names():
    path1 = os.path.join(GIFS_DIR, f"{topo_name}_blood_flow.gif")
    path2 = os.path.join(GIFS_DIR, f"{topo_name}_pinn_prediction.gif")
    out_path = os.path.join(GIFS_DIR, f"{topo_name}_comparison.gif")

    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"Skipping {topo_name}: missing GIF(s)")
        continue

    print(f"\nProcessing {topo_name}...")
    frames1 = iio.imread(path1, index=None)
    frames2 = iio.imread(path2, index=None)
    min_len = min(len(frames1), len(frames2))
    print(f"  MUSCL: {len(frames1)} frames, PINN: {len(frames2)} frames -> {min_len}")

    new_frames = []
    for i in range(min_len):
        im1 = Image.fromarray(frames1[i])
        im2 = Image.fromarray(frames2[i])
        w = im1.width + im2.width
        h = max(im1.height, im2.height)
        new_im = Image.new('RGB', (w, h), (30, 30, 30))
        new_im.paste(im1, (0, 0))
        new_im.paste(im2, (im1.width, 0))
        draw = ImageDraw.Draw(new_im)
        font = ImageFont.load_default()

        title = topo_name.replace("_", " ").title()
        text1 = "MUSCL (Simulation)"
        text2 = "PINN (Prediction)"

        # Title centered at top
        bb_t = draw.textbbox((0,0), title, font=font)
        tw_t = bb_t[2] - bb_t[0]
        for off in [(0,0),(1,0),(0,1),(1,1)]:
            draw.text((w//2 - tw_t//2 + off[0], 5 + off[1]), title, fill=(255,200,50), font=font)

        bb1 = draw.textbbox((0,0), text1, font=font)
        bb2 = draw.textbbox((0,0), text2, font=font)
        tw1 = bb1[2]-bb1[0]; tw2 = bb2[2]-bb2[0]
        for off in [(0,0),(1,0),(0,1),(1,1)]:
            draw.text((im1.width//2-tw1//2+off[0], 22+off[1]), text1, fill=(255,255,255), font=font)
            draw.text((im1.width+im2.width//2-tw2//2+off[0], 22+off[1]), text2, fill=(255,255,255), font=font)

        new_frames.append(np.array(new_im))

    iio.imwrite(out_path, new_frames, plugin="pillow", duration=1000/25, loop=0)
    print(f"  Saved {out_path}")

print("\nAll comparisons done!")
