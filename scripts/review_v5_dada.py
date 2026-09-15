"""Create numbered diagnostic contact sheets without model predictions."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/v5_external/dada'
OUT = BASE/'blind_review'
OUT.mkdir(exist_ok=True)
for report_path in sorted((BASE/'extraction_reports').glob('*.json')):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    key = report['ID']
    if all((OUT/f'{key}_{kind}.jpg').exists() for kind in ('overview','contact')):
        continue
    paths = sorted((BASE/'inputs/images'/key).glob('*.png'))
    accident = report['source']['provided_accident_frame']
    groups = {
        'overview': sorted(set(np.linspace(1,len(paths),16).round().astype(int))),
        'contact': sorted(set(max(1,min(len(paths),accident+d)) for d in [-30,-20,-12,-9,-6,-3,-1,0,1,3,6,9,12,16,22,30])),
    }
    for kind, numbers in groups.items():
        sheet = Image.new('RGB',(1600,4*250),'white')
        draw = ImageDraw.Draw(sheet)
        for i,n in enumerate(numbers):
            im = Image.open(paths[n-1]).convert('RGB')
            im.thumbnail((400,225))
            x,y=(i%4)*400,(i//4)*250
            sheet.paste(im,(x+(400-im.width)//2,y+25))
            draw.text((x+5,y+5),f'{key} | ORIGINAL FRAME {n}',fill='black')
        sheet.save(OUT/f'{key}_{kind}.jpg',quality=95)
    print(key, len(paths), flush=True)
