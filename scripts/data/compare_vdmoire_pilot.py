"""Small source-group-separated recapture diagnostic using existing patch features."""
import hashlib,json,sys
from pathlib import Path
import cv2,joblib,numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from solution.stage1 import _patch_features
OUT=ROOT/'artifacts/data_pilot_20260916/vdmoire'

def main():
    raw=json.loads((OUT/'manifest.json').read_text())['frames']
    clean=json.loads((OUT/'clean_manifest.json').read_text())['frames']
    assert len(clean)==40 and len(raw)==120,'Wait for complete clean acquisition'
    groups=sorted({r['source_group'] for r in raw},key=lambda g:hashlib.sha256(('vd-pilot:'+g).encode()).hexdigest())
    held=set(groups[:5]);x=[];y=[];test=[];records=[]
    cv2.setNumThreads(2)
    for group in groups:
        for label,items in [(0,clean),(1,raw)]:
            paths=sorted(r['path'] for r in items if r['source_group']==group)
            # Equal two-frame aggregation for each class; no invented temporal matching.
            selected=[paths[0],paths[-1]];features=[]
            for p in selected:
                with Image.open(ROOT/p) as im:
                    a=np.array(im.convert('RGB'));h,w=a.shape[:2]
                    patch=a[h//2-96:h//2+96,w//2-96:w//2+96]
                    assert patch.shape==(192,192,3)
                    features.append(_patch_features(patch))
            v=np.median(features,axis=0);v[13]=np.log1p(v[13])
            x.append(v);y.append(label);test.append(group in held)
            records.append({'source_group':group,'label':label,'paths':selected,'split':'development_validation' if group in held else 'train'})
    x=np.array(x);y=np.array(y);test=np.array(test)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=42))
    model.fit(x[~test],y[~test]);pred=model.predict(x[test])
    report={'scope':'20-source tiny pilot, single iPhone device. Existing patch features only; not current full Stage1 or road-domain validation.',
            'clean_source':'provider targets; group mapping verified by filename formula, pixel/time matching not assumed',
            'preprocessing':'two frames per class/source, central native 192px patch, 19 existing forensic features, median aggregation; no display-border features',
            'train_groups':15,'validation_groups':5,'macro_f1':float(f1_score(y[test],pred,average='macro')),
            'confusion_original_recaptured':confusion_matrix(y[test],pred,labels=[0,1]).tolist(),
            'validation_predictions':[dict(records[i],prediction=int(p)) for i,p in zip(np.flatnonzero(test),pred)],
            'decision':'Do not replace full Stage1 based on this pilot. UHDM increment and independent road evaluation remain pending.'}
    (OUT/'split_and_inputs.json').write_text(json.dumps(records,indent=2))
    (OUT/'pilot_result.json').write_text(json.dumps(report,indent=2));joblib.dump(model,OUT/'patch_pilot.joblib')
    print(json.dumps({k:report[k] for k in ['macro_f1','confusion_original_recaptured']}))
if __name__=='__main__':main()
