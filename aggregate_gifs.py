import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

print("Wczytywanie animacji...")
frames1 = iio.imread("blood_flow.gif", index=None)
frames2 = iio.imread("pinn_prediction.gif", index=None)

min_len = min(len(frames1), len(frames2))
print(f"blood_flow.gif: {len(frames1)} frames")
print(f"pinn_prediction.gif: {len(frames2)} frames")
print(f"Generowanie porównania ({min_len} klatek)...")

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
    
    text1 = "MUSCL (Simulation)"
    text2 = "PINN (Prediction)"
    
    bbox1 = draw.textbbox((0, 0), text1, font=font)
    tw1 = bbox1[2] - bbox1[0]
    bbox2 = draw.textbbox((0, 0), text2, font=font)
    tw2 = bbox2[2] - bbox2[0]
    
    # Scale up the font if we can by using a bigger standard font, or just draw multiple times to make it bold/bigger (this is just default)
    # Using a simple trick to make it more visible:
    for offset in [(0,0), (1,0), (0,1), (1,1)]:
        draw.text((im1.width // 2 - tw1 // 2 + offset[0], 20 + offset[1]), text1, fill=(255, 255, 255), font=font)
        draw.text((im1.width + im2.width // 2 - tw2 // 2 + offset[0], 20 + offset[1]), text2, fill=(255, 255, 255), font=font)
    
    new_frames.append(np.array(new_im))

print("Zapisywanie comparison.gif...")
iio.imwrite("comparison.gif", new_frames, plugin="pillow", duration=1000/25, loop=0)
print("Gotowe!")
