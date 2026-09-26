"""Small source-frame evidence sheets and sensor plots using installed Pillow."""
from compare import O,R,Z,B,read,write,load,ACC
from pathlib import Path
import numpy as np,av
from PIL import Image,ImageDraw,ImageFont
cs=load();pairs=read(O/'matched_manifest.json');results={r['pair_id']:r for r in read(O/'results.json')};out=O/'evidence';out.mkdir(exist_ok=True)
fp='/System/Library/Fonts/Supplemental/Arial.ttf';font=ImageFont.truetype(fp,17) if Path(fp).exists() else ImageFont.load_default()
requests={};images={};sources=[]
for p in pairs:
 for name in ['comma','zod']:
  q=p[name];requests.setdefault(q['key'],set()).update([q['i']-5,q['i'],q['i']+5])
for key,wanted in requests.items():
 c=cs[key];d=c['d']
 if c['dataset']=='comma':
  frame_to_sample={int(d['frame_index'][k]):k for k in wanted};path=R/c['row']['raw_path']
  with av.open(str(path)) as con:
   for j,f in enumerate(con.decode(video=0)):
    if j in frame_to_sample:
     k=frame_to_sample[j];images[key,k]=f.to_image().resize((384,216));sources.append({'dataset':'comma','id':c['row']['id'],'sample_index':k,'native_frame_index':j,'frame_time':float(d['frame_time'][k]),'path':str(path.relative_to(R)),'pts':f.pts,'pts_note':'frame_times used for sensor alignment; HEVC may lack PTS'})
    if j>=max(frame_to_sample):break
 else:
  frames=read(Z/'raw/sequences'/c['row']['id']/'info.json')['camera_frames']['front_blur']
  for k in wanted:
   fi=int(d['frame_index'][k]);path=Z/'raw'/frames[fi]['filepath'];images[key,k]=Image.open(path).resize((384,216));sources.append({'dataset':'zod','id':c['row']['id'],'sample_index':k,'native_frame_index':fi,'frame_time':float(d['frame_time'][k]),'path':str(path.relative_to(R))})
assert len(images)==sum(len(x) for x in requests.values())
def plot(draw,box,series,title):
 x0,y0,x1,y1=box;x0+=65;y0+=32;y1-=30
 vals=np.concatenate([s[1][np.isfinite(s[1])] for s in series]);lo,hi=float(vals.min()),float(vals.max());pad=max((hi-lo)*.12,.03);lo-=pad;hi+=pad
 draw.text((x0,y0-26),title,font=font,fill='black');draw.rectangle((x0,y0,x1,y1),outline='#999999')
 for tick in np.linspace(lo,hi,5):
  y=y1-(tick-lo)/(hi-lo)*(y1-y0);draw.line((x0,y,x1,y),fill='#dddddd');draw.text((x0-63,y-9),f'{tick:.2f}',font=font,fill='black')
 for tick in range(-3,4):
  x=x0+(tick+3)/6*(x1-x0);draw.text((x-8,y1+4),str(tick),font=font,fill='black')
 for time,val,color in series:
  prev=None
  for t,v in zip(time,val):
   if not np.isfinite(v):prev=None;continue
   pt=(x0+(t+3)/6*(x1-x0),y1-(v-lo)/(hi-lo)*(y1-y0))
   if prev is not None:draw.line((prev,pt),fill=color,width=3)
   prev=pt
 for tick in [-.5,.5]:
  x=x0+(tick+3)/6*(x1-x0);draw.line((x,y0,x,y1),fill='#999999',width=1)
for p in pairs:
 sheet=Image.new('RGB',(1152,610),'white');draw=ImageDraw.Draw(sheet)
 draw.text((8,4),f"{p['pair_id']} {ACC[p['truth']]} | sensor selected, not feature/prediction selected",font=font,fill='black')
 for row,name in enumerate(['comma','zod']):
  q=p[name];d=cs[q['key']]['d'];i=q['i'];yp=35+row*285
  draw.text((8,yp),f"{q['key']} t={i/10:.1f}s | speed={q['v']:.3f}m/s accel={q['a']:+.3f}m/s2",font=font,fill='black')
  for col,k in enumerate([i-5,i,i+5]):
   sheet.paste(images[q['key'],k],(col*384,yp+48));draw.text((col*384+4,yp+24),f"t={k/10:.1f}s frame={int(d['frame_index'][k])}",font=font,fill='black')
 sheet.save(out/(p['pair_id']+'_frames.jpg'))
 plotimage=Image.new('RGB',(1200,650),'white');draw=ImageDraw.Draw(plotimage)
 draw.text((15,5),f"{p['pair_id']} Blue: comma / Orange: ZOD; central1s matched; context6s matched={p['context6s_matched']}",font=font,fill='black')
 vv=[];aa=[]
 for name,color in [('comma','#1764aa'),('zod','#d56117')]:
  q=p[name];d=cs[q['key']]['d'];i=q['i'];t=np.arange(-30,31)/10
  vv.append((t,d['speed'][i-30:i+31],color));aa.append((t,d['acceleration_proxy'][i-30:i+31],color))
 plot(draw,(0,30,590,330),vv,'Raw speed (m/s)');plot(draw,(600,30,1190,330),aa,'Existing1s slope (m/s2)')
 r=results[p['pair_id']];blocks=r['A_minus_D_contribution']['by_block'];maxv=max(1,max(abs(x) for x in blocks));names=['raw','mean0.5s','mean1.5s','mean3.1s','diff1s','diff3s']
 draw.text((15,345),'Change in fixed comma-model A-D margin: ZOD minus comma (exact linear attribution)',font=font,fill='black')
 for j,(label,value) in enumerate(zip(names,blocks)):
  y=380+j*39;center=575;end=center+value/maxv*350;draw.text((25,y),label,font=font,fill='black');draw.line((center,y-2,center,y+27),fill='#333333');draw.rectangle((min(center,end),y,max(center,end),y+23),fill='#d56117' if value>0 else '#1764aa');draw.text((970,y),f'{value:+.3f}',font=font,fill='black')
 plotimage.save(out/(p['pair_id']+'_sensors_contribution.png'))
write(out/'source_frames.json',sources);print('evidence sheets',len(pairs),'source frames',len(images),flush=True)
