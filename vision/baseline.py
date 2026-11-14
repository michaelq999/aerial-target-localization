"""Untrained classical computer-vision baseline: green segmentation + centroid.
Use it to verify pipeline before CNN training; it may fail on green grass.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def detect(image_bgr):
    hsv=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
    # Empirical OpenCV HSV thresholds; tune using real camera samples.
    mask=cv2.inRange(hsv,np.array([34,65,35],np.uint8),np.array([88,255,255],np.uint8))
    # White board near green: use proximity to white to suppress surrounding green ground.
    white=cv2.inRange(hsv,np.array([0,0,155],np.uint8),np.array([179,90,255],np.uint8))
    white=cv2.dilate(white,np.ones((13,13),np.uint8))
    mask=cv2.bitwise_and(mask,white)
    if cv2.countNonZero(mask)<12: return None,mask
    # Pick the largest connected green region. The two crossing bars should connect.
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    if count<2:return None,mask
    index=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    x,y=centroids[index]
    return (float(x),float(y)),mask


def main():
    p=argparse.ArgumentParser()
    p.add_argument('image')
    p.add_argument('--out',default='output/baseline_result.png')
    a=p.parse_args()
    img=cv2.imread(a.image)
    if img is None:raise FileNotFoundError(a.image)
    center,mask=detect(img)
    if center is not None:
        x,y=map(round,center)
        cv2.drawMarker(img,(x,y),(0,0,255),cv2.MARKER_CROSS,25,2)
        print(f'Baseline center (x,y): ({center[0]:.1f}, {center[1]:.1f})')
    else: print('Baseline found no green X candidate')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out),img)
    cv2.imwrite(str(out.with_name(out.stem+'_mask.png')),mask)
    print('Wrote',out)

if __name__=='__main__':main()
