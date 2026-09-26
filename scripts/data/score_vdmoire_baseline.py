"""Score the unchanged Mac Stage1 ensemble on the same pilot source groups."""
import json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from solution.stage1 import extract_features,feature_probability
from solution.stage1_tpo_merged import TPODetector
OUT=ROOT/'artifacts/data_pilot_20260916/vdmoire'
MODEL=ROOT/'artifacts/submissions/verify_v6/model/stage1'

def main():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    cfg=json.loads((MODEL/'config.json').read_text());artifact=json.loads((MODEL/cfg['forensic_artifact']).read_text())
    records=[r for r in json.loads((OUT/'split_and_inputs.json').read_text()) if r['split']=='development_validation']
    detector=TPODetector(MODEL/'tpo');result={};start=time.monotonic()
    try:
        for mode in ['full_frame','central_192']:
            rows=[]
            for r in records:
                frames=[]
                for p in r['paths']:
                    with Image.open(ROOT/p) as im:
                        a=np.array(im.convert('RGB'))
                        if mode=='central_192':
                            h,w=a.shape[:2];a=a[h//2-96:h//2+96,w//2-96:w//2+96]
                        frames.append(a)
                feat,_=extract_features(frames);p=feature_probability(feat,artifact);t=float(detector.score(frames).mean());w=cfg['forensic_weight'];score=w*p+(1-w)*t
                rows.append(dict(r,forensic_probability=p,tpo_probability=t,probability=score,prediction=int(score>=cfg['threshold'])))
            y=[r['label'] for r in rows];pred=[r['prediction'] for r in rows]
            result[mode]={'macro_f1':float(f1_score(y,pred,average='macro')),'confusion':confusion_matrix(y,pred,labels=[0,1]).tolist(),'rows':rows}
            print(mode,result[mode]['macro_f1'],flush=True)
    finally:detector.close()
    result['scope']='Unchanged ensemble weights, two sampled frames per class/source. Full-frame includes display border; central crop changes input distribution. Neither is independent road-video performance.'
    result['seconds']=time.monotonic()-start
    result['new_patch_pilot']=json.loads((OUT/'pilot_result.json').read_text())['macro_f1']
    (OUT/'baseline_comparison.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
