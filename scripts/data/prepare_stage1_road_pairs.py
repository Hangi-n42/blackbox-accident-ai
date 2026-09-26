"""Match camera source to physical recapture; crop without perspective resampling."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from build_stage1_road_pairs import ROOT,OUT,write
cv2.setNumThreads(2)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jpeg95(a):
 ok,b=cv2.imencode('.jpg',a,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
 return cv2.imdecode(b,cv2.IMREAD_COLOR)
def match(a,b):
 sift=cv2.SIFT_create(nfeatures=4000)
 ka,da=sift.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None);kb,db=sift.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
 if da is None or db is None:return None,{'inliers':0}
 pairs=cv2.BFMatcher().knnMatch(da,db,k=2);good=[p[0] for p in pairs if len(p)==2 and p[0].distance<.72*p[1].distance]
 if len(good)<12:return None,{'inliers':0,'matches':len(good)}
 src=np.float32([ka[m.queryIdx].pt for m in good]);dst=np.float32([kb[m.trainIdx].pt for m in good]);h,mask=cv2.findHomography(src,dst,cv2.RANSAC,3)
 if h is None:return None,{'inliers':0}
 sel=mask.ravel().astype(bool);err=np.linalg.norm(cv2.perspectiveTransform(src[:,None],h)[:,0]-dst,axis=1)
 meta={'inliers':int(sel.sum()),'matches':len(good),'inlier_ratio':float(sel.mean()),'median_error':float(np.median(err[sel])),'homography':h.tolist()}
 return h,meta

def prepare():
 protocol=json.loads((OUT/'selection.json').read_text());rows=[];qa=[];issues=[]
 for split,numbers in protocol['selection'].items():
  for n in numbers:
   originals=sorted(ROOT/r['path'] for r in json.loads((OUT/'reds_frames.json').read_text()) if Path(r['path']).parent.name==f'{n:03d}');captures=sorted(ROOT/r['path'] for r in json.loads((OUT/'vd_frames.json').read_text()) if Path(r['path']).parent.name==f'{n:03d}')
   assert len(originals)==len(captures)==12,(n,len(originals),len(captures))
   for i,(op,rp) in enumerate(zip(originals,captures)):
    a=cv2.imread(str(op));b=cv2.imread(str(rp));h,m=match(a,b)
    record={'source':n,'frame':i,'original_member_frame':op.name,'recapture_member_frame':rp.name,**m}
    if h is None or m['inliers']<12 or m['inlier_ratio']<.2 or m['median_error']>3:
     issues.append(record);qa.append(record);continue
    ah,aw=a.shape[:2];bh,bw=b.shape[:2]
    polygon=cv2.perspectiveTransform(np.float32([[[0,0],[aw-1,0],[aw-1,ah-1],[0,ah-1]]]),h)[0]
    center=cv2.perspectiveTransform(np.float32([[[aw/2,ah/2]]]),h)[0,0]
    # A fixed native 640x360 window strictly inside the displayed content.
    cx,cy=map(float,center);x0,y0=round(cx-320),round(cy-180);x1,y1=x0+640,y0+360
    corners=np.float32([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]])
    safe=min(cv2.pointPolygonTest(polygon,tuple(map(float,p)),True) for p in corners)
    if safe<10 or x0<0 or y0<0 or x1>bw or y1>bh:
     record['crop_failure']={'margin':safe,'rect':[x0,y0,x1,y1]};issues.append(record);qa.append(record);continue
    inv=cv2.perspectiveTransform(corners[None],np.linalg.inv(h))[0]
    ox0,oy0=np.floor(inv.min(0)).astype(int);ox1,oy1=np.ceil(inv.max(0)).astype(int)+1
    if ox0<0 or oy0<0 or ox1>aw or oy1>ah:issues.append(record);qa.append(record);continue
    record.update({'recapture_rect':[x0,y0,x1,y1],'original_rect':list(map(int,[ox0,oy0,ox1,oy1])),'screen_margin_px':safe,'polygon':polygon.tolist(),'status':'pass_geometric_crop_gate'})
    qa.append(record)
    for label,src,rgb in [('original',op,a[oy0:oy1,ox0:ox1]),('recapture',rp,b[y0:y1,x0:x1])]:
     frame=jpeg95(cv2.resize(rgb,(640,360),interpolation=cv2.INTER_AREA))
     p=OUT/'prepared'/split/f'{n:03d}'/label/f'{i:02d}.png';p.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(p),frame)
     rows.append({'source':n,'source_group':f'REDS_train_{n:03d}','cluster':protocol['location_clusters'][str(n)],'split':split,'label':label,'frame':i,'path':str(p.relative_to(ROOT)),'sha256':digest(p),'parent':str(src.relative_to(ROOT)),'parent_sha256':digest(src)})
   print('prepared',n,'issues',sum(x['source']==n for x in issues),flush=True)
 write(OUT/'crop_qa.json',qa);write(OUT/'preparation_issues.json',issues);write(OUT/'prepared_frames.json',rows)
 for split,numbers in protocol['selection'].items():
  for page,portion in enumerate([numbers[:4],numbers[4:]]):
   im=Image.new('RGB',(1280,720),'#222');d=ImageDraw.Draw(im)
   for j,n in enumerate(portion):
    for k,label in enumerate(['original','recapture']):
     p=OUT/'prepared'/split/f'{n:03d}'/label/'06.png'
     if p.exists():
      pic=Image.open(p).resize((320,180));im.paste(pic,((j%2)*640+k*320,(j//2)*360+25));d.text(((j%2)*640+k*320,(j//2)*360),f'{n:03d} {label}',fill='white')
   im.save(OUT/f'qa_{split}_{page}.jpg')
 print('rows',len(rows),'issues',len(issues),flush=True)
if __name__=='__main__':prepare()
