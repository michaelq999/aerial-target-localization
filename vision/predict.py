"""Locate green X center and write a visualization of the heatmap."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from model import CenterHeatmapNet,spatial_probs,location_from_probs


def run(weights,image_path,out):
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    try: checkpoint=torch.load(weights,map_location=device,weights_only=True)
    except TypeError: checkpoint=torch.load(weights,map_location=device)
    model=CenterHeatmapNet().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    size=checkpoint['trained_image_size']
    with Image.open(image_path) as inp:
        full=inp.convert('RGB')
    original_w,original_h=full.size
    cropped=full.resize((size[1],size[0]),Image.Resampling.BILINEAR)
    arr=np.asarray(cropped,dtype=np.float32).transpose(2,0,1).copy()/255.0
    with torch.no_grad():
        p=spatial_probs(model(torch.from_numpy(arr).unsqueeze(0).to(device)))[0].cpu()
    x,y,peak=location_from_probs(p)
    # Interpret coordinates in original image, not resized network input.
    orig_x=x * original_w / size[1]
    orig_y=y * original_h / size[0]
    heat=p.numpy()
    # Heatmap is scaled relatively for display; peak probability is also reported separately.
    scaled=np.clip((heat/max(float(heat.max()),1e-12))**0.45*255,0,255).astype('uint8')
    layer=np.zeros((size[0],size[1],3),dtype='uint8')
    layer[:,:,0]=scaled
    layer[:,:,1]=((255-scaled)*scaled/255).astype('uint8')
    overlay=Image.blend(cropped,Image.fromarray(layer),alpha=0.32)
    draw=ImageDraw.Draw(overlay)
    draw.ellipse((x-6,y-6,x+6,y+6),outline=(255,255,255),width=2)
    draw.line((x-9,y,x+9,y),fill=(0,0,0),width=2)
    draw.line((x,y-9,x,y+9),fill=(0,0,0),width=2)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    overlay.save(out)
    print(f'Predicted center original-image pixels: ({orig_x:.1f}, {orig_y:.1f})')
    print(f'Heatmap peak probability per pixel: {peak:.6f} (not an object-detection confidence score)')
    print(f'Heatmap overlay: {out}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('image')
    p.add_argument('--weights',default='output/green_x_model.pt')
    p.add_argument('--out',default='output/prediction_overlay.png')
    a=p.parse_args()
    run(a.weights,a.image,a.out)

if __name__=='__main__':main()
