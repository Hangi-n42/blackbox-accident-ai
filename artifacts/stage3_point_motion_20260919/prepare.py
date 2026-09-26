"""Sensor-only bounded pair selection before viewing candidate features."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
import cv2,av
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'artifacts/stage3_training_basis_20260917'
read=lambda p:json.loads(p.read_text())
write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    assert not (O/'freeze.json').exists()
    cs={c['id']:c for c in read(B/'cases.json')}
    split=read(R/'artifacts/stage3_mixed_datecheck_20260917/fold_1/split_manifest.json')
    train={r['id'] for r in split if r['experiment_role']=='train'}
    rows=[]
    for id,c in cs.items():
        z=np.load(B/c['labels_npz'])
        for i in range(15,len(z['time'])-15,10):
            w=slice(i-10,i+11);a=z['acceleration_proxy'][w];v=z['speed_smoothed'][w];s=z['steering_smoothed'][w]
            if not np.isfinite(a).all() or not np.isfinite(v).all() or not np.isfinite(s).all() or min(v)<5 or max(abs(s))>5:continue
            k=0 if min(a)>.5 else 1 if max(a)<-.5 else 2 if max(abs(a))<.1 else -1
            if k<0:continue
            rows.append({'key':f'{id}_{i}','id':id,'index':i,'role':'train' if id in train else 'held',
                         'vehicle':c['vehicle'],'route':c['route'],'label':k,'v':float(v[10]),'a':float(a[10]),
                         'max_abs_steer':float(max(abs(s))),'sensor_ratio_a_v':float(a[10]/v[10]),
                         'raw_path':c['raw_path'],'labels_npz':c['labels_npz'],'source_url':c['source_url'],'license':c['license']})
    selected={};pairs=[];availability=[]
    for role,vehicle in itertools.product(['train','held'],sorted({r['vehicle'] for r in rows})):
        g=[r for r in rows if r['role']==role and r['vehicle']==vehicle]
        availability.append({'role':role,'vehicle':vehicle,'candidate_counts':[sum(r['label']==k for r in g) for k in range(3)]})
        for ka,kb in [(0,1),(2,0),(2,1)]:
            candidates=sorted((abs(a['v']-b['v']),a['key'],b['key']) for a in g for b in g if a['label']==ka and b['label']==kb and abs(a['v']-b['v'])<=2)
            taken=[]
            lookup={r['key']:r for r in g}
            for dv,ak,bk in candidates:
                a,b=lookup[ak],lookup[bk]
                if any(r['id']==s['id'] and abs(r['index']-s['index'])<30 for r in [a,b] for s in taken):continue
                if a['id']==b['id'] and abs(a['index']-b['index'])<30:continue
                selected[ak]=a;selected[bk]=b;taken.extend([a,b])
                pairs.append({'pair':len(pairs),'role':role,'vehicle':vehicle,'classes':[ka,kb],'a_key':ak,'b_key':bk,'speed_delta':dv})
                if len(taken)>=4:break
        # Two non-overlapping constant controls per vehicle/split, speed extremes.
        const=sorted([r for r in g if r['label']==2],key=lambda r:(r['v'],r['key']))
        for r in (const[:1]+const[-1:]):selected[r['key']]=r
    protected=read(R/'artifacts/stage3_official_bias_20260919/freeze.json')['paths']
    assert all(sha(R/p)==h for p,h in protected.items())
    write(O/'freeze.json',{'prepare_sha256':sha(Path(__file__)),'source_inputs_protected':protected,
        'selection':'existing date1 source split; 21 samples /2sec windows center stride1sec; min speed>5m/s, |steer|<=5degrees, all21 acceleration>.5 or <-.5 or |a|<.1; samevehicle/split pair speedcaliper2m/s; greedily min speedgap max2 nonoverlap pairs per classpair; add two constant speed extremes per vehicle/split; no prediction or visual feature used',
        'feature':'640width nativeframes aligned to saved frame_index/frame_time; LK21x21 pyramid3 forward/backward<=1px, photometric err<=20; max500 corners minDistance8; full21frame tracks; robust flow-line FOE RANSAC residual3px then IRLS; retain lines residual<=3px, r>=25px, expanding; quadratic 1/r normalized RMSE<=.1, slope<-1e-5; scalar median u_second/u_first, ideal straight static translation gives a/v; raw radius second/first as confounded control; no rotation correction',
        'quality':'atleast15 usable tracks, FOE normalizedx.15-.85/y.15-.70, left/right track support>=3 each, firsthalf/secondhalf FOE drift<=20px; manual static road/background rectangles reviewed before feature values; manual mask is diagnostic assistance, not automated deployable support',
        'gates':'training usable>=2 per A/D/C and >=2 source routes; freeze deadband from max(.01, trainconstant90percentile absolute feature); require threshold <.05/s. Held coverage>=.6 each A/D/C; >=3 held windows per A/D/C from>=2heldroutes overall; >=.8heldconstant inside deadband; >=.7heldA/D correct sign and outside deadband perclass; >=.75expected pair ordering; heldconstant medianabs candidate <raw control. If insufficient support fail readiness, not proof method impossible. No tuning after result; only pass permits classifier addition.',
        'limitations':'small previously development-exposed23sources; labels are speed-derivative proxy not official; source split does not imply new independent test; steering selection is not calibrated camera rotation; manual static masks can include unrecognized movers; feature missing never zero; no absolute acceleration inference',
        'opencv_reference':'https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html',
        'availability':availability,'pairs':pairs})
    write(O/'windows.json',list(selected.values()));pd.DataFrame(rows).to_csv(O/'eligible_sensor_windows.csv',index=False);pd.DataFrame(pairs).to_csv(O/'pairs.csv',index=False)
    panels=[]
    for id in sorted({r['id'] for r in selected.values()}):
        c=cs[id];z=np.load(B/c['labels_npz']);windows=[r for r in selected.values() if r['id']==id]
        needed={j for r in windows for j in range(r['index']-10,r['index']+11)}
        native_to_i={int(z['frame_index'][i]):i for i in needed};images={};actual={}
        with av.open(str(R/c['raw_path'])) as con:
            for native,frame in enumerate(con.decode(video=0)):
                if native in native_to_i:
                    i=native_to_i[native];bgr=frame.to_ndarray(format='bgr24');h=round(bgr.shape[0]*640/bgr.shape[1]);images[i]=cv2.resize(bgr,(640,h));actual[i]=float(frame.time) if frame.time is not None else None
                if native>=max(native_to_i):break
        assert set(images)==needed
        for r in windows:
            ix=np.arange(r['index']-10,r['index']+11);rgb=np.stack([cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB) for i in ix]);t=z['frame_time'][ix].astype(float);assert np.all(np.diff(t)>0)
            np.savez_compressed(O/(r['key']+'_frames.npz'),rgb=rgb,time=t,frame_index=z['frame_index'][ix],decoder_time=np.array([actual[i] for i in ix],dtype=float))
            h=rgb.shape[1];im=Image.new('RGB',(960,round(h*.5)+46),'white');draw=ImageDraw.Draw(im)
            draw.text((5,3),f"{r['key']} {r['role']} class{r['label']} v={r['v']:.2f} a={r['a']:.3f}",fill='black')
            for col,k in enumerate([0,10,20]):im.paste(Image.fromarray(rgb[k]).resize((320,round(h*.5))),(320*col,40))
            panels.append((r['key'],im))
        print(id,len(windows),'windows decoded',flush=True)
    for start in range(0,len(panels),6):
        page=panels[start:start+6];canvas=Image.new('RGB',(960,sum(im.height for _,im in page)),'white');y=0
        for _,im in page:canvas.paste(im,(0,y));y+=im.height
        canvas.save(O/f'review_{start//6}.jpg')
    write(O/'review_pages.json',[[key for key,_ in panels[i:i+6]] for i in range(0,len(panels),6)])
    print('selected',len(selected),'pairs',len(pairs),flush=True)

if __name__=='__main__':
    cv2.setNumThreads(2);main()
