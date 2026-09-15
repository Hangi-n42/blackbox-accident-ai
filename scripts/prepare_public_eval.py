"""Convert only released public examples to the official inference layout."""
from pathlib import Path
import shutil
import cv2
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'Baseline/data'
TARGET=ROOT/'artifacts/public_eval'

def main():
    for stage in ('stage1','stage3'):
        (TARGET/stage/'videos').mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv(SOURCE/'stage1/labels.csv')
    for row in labels.itertuples():
        source=SOURCE/'stage1'/row.path
        shutil.copy2(source,TARGET/'stage1/videos'/f'{row.ID}{source.suffix}')
    labels=pd.read_csv(SOURCE/'stage2/labels.csv')
    for row in labels.itertuples():
        folder=TARGET/'stage2/images'/row.ID
        folder.mkdir(parents=True,exist_ok=True)
        cap=cv2.VideoCapture(str(SOURCE/'stage2'/row.path));index=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            out=folder/f'frame_{index:06d}.jpg'
            if not out.exists():
                ok=cv2.imencode('.jpg',frame)[1].tofile(str(out))
            index+=1
        cap.release()
        print(row.ID,index,flush=True)
    for source in (SOURCE/'stage3/videos').glob('*.mp4'):
        shutil.copy2(source,TARGET/'stage3/videos'/source.name)
    print(TARGET,flush=True)

if __name__=='__main__':main()
