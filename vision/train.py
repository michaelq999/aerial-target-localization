"""Train a synthetic-image-to-center-probability-heatmap model."""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Subset
from model import CenterHeatmapNet, spatial_probs, location_from_probs


class GreenXDataset(Dataset):
    def __init__(self,dataset_dir,sigma=3.0):
        self.root=Path(dataset_dir)
        with (self.root/'labels.csv').open() as f:
            self.rows=list(csv.DictReader(f))
        if not self.rows: raise ValueError('No labels in dataset')
        self.sigma=sigma

    def __len__(self):return len(self.rows)

    def __getitem__(self,idx):
        row=self.rows[idx]
        with Image.open(self.root/'images'/row['filename']) as im:
            a=np.asarray(im.convert('RGB'),dtype=np.float32)/255.0
        h,w=a.shape[:2]
        x,y=float(row['x']),float(row['y'])
        xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
        target=np.exp(-((xx-x)**2+(yy-y)**2)/(2*self.sigma**2)).astype(np.float32)
        target/=target.sum()
        return torch.from_numpy(a.transpose(2,0,1).copy()),torch.from_numpy(target),torch.tensor([x,y],dtype=torch.float32)


def validate(model,loader,device):
    model.eval()
    errors=[]
    with torch.no_grad():
        for images,_,coords in loader:
            preds=spatial_probs(model(images.to(device))).cpu()
            for p,gt in zip(preds,coords):
                x,y,_=location_from_probs(p)
                errors.append(math.hypot(x-float(gt[0]),y-float(gt[1])))
    return float(np.mean(errors)) if errors else float('nan')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='output/dataset')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--sigma',type=float,default=3.0)
    p.add_argument('--out',default='output/green_x_model.pt')
    p.add_argument('--seed',type=int,default=7)
    args=p.parse_args()
    if args.epochs<=0: raise ValueError('epochs must be positive')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    data=GreenXDataset(args.dataset,args.sigma)
    if len(data)<5: raise ValueError('Use at least 5 images, ideally 200+')
    indices=list(range(len(data)));random.shuffle(indices)
    n_val=max(1,int(.2*len(indices)))
    train_set=Subset(data,indices[n_val:])
    val_set=Subset(data,indices[:n_val])
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_set,batch_size=args.batch_size,shuffle=False,num_workers=0)
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model=CenterHeatmapNet().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    (out.parent/'validation_files.txt').write_text('\n'.join(data.rows[i]['filename'] for i in indices[:n_val])+'\n')
    best=float('inf')
    print(f'Training {len(train_set)} images, validating {len(val_set)}, device={device}')
    for epoch in range(args.epochs):
        model.train();losses=[]
        for images,targets,_ in train_loader:
            images=images.to(device);targets=targets.to(device)
            logits=model(images)
            # Target is a normalized Gaussian centered at (x,y).
            # Cross entropy trains a spatial probability distribution, not a single coordinate.
            log_p=torch.log_softmax(logits.flatten(1),dim=1)
            loss=-(targets.flatten(1)*log_p).sum(dim=1).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            losses.append(float(loss.item()))
        error=validate(model,val_loader,device)
        print(f'Epoch {epoch+1:02d}/{args.epochs}: train cross-entropy={np.mean(losses):.3f}, val mean center error={error:.2f} px')
        if error<best:
            best=error
            torch.save({'model_state':{k:v.cpu() for k,v in model.state_dict().items()},
                        'sigma':args.sigma,'trained_image_size':list(data[0][0].shape[-2:])},out)
    print(f'Best validation mean center error: {best:.2f} px; saved {out}')
    print('NOTE: This model always proposes a center; no-target rejection needs negative examples.')

if __name__=='__main__':main()
