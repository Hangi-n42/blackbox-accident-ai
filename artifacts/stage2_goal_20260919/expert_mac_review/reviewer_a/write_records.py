import json,hashlib
from pathlib import Path
B=Path('/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919')
O=B/'expert_mac_review/reviewer_a'; shared=B/'expert_mac_review/blind_packet'
M={v:json.loads((B/'acquisition'/f'{v}.pts.json').read_text()) for v in ['00024','00025','00026','00027','00028','00029']}
def point(v,f): return dict(M[v]['mapping'][f])
def interval(v,a,b): return {'start':point(v,a),'end':point(v,b),'bounds':'inclusive review interval; not an exact physical-event bracket'}
def ev(v,tag,f,standalone=False):
 d=O/v/tag; fs=json.loads((d/'frames.json').read_text()); ids=[x['frame_id'] for x in fs]
 k=ids.index(f); page=18 if tag=='full_2hz' else 12
 e={'image_path':str(d/f'sheet_{k//page+1:02}.jpg'),'frame':point(v,f),'view_method':'contact_sheet'}
 if standalone: e={'image_path':str(d/f'f{f:04}.png'),'frame':point(v,f),'view_method':'native_resolution_png'}
 return e
def se(v,f):
 j=json.loads((shared/v/'manifest.json').read_text());ids=[x['frame'] for x in j['frames']];k=ids.index(f)
 return {'image_path':str(shared/v/f'sheet_{k//16:02}.jpg'),'frame':point(v,f),'view_method':'contact_sheet'}
def item(status,value,reason,evidence,**kw):
 return {'status':status,'value':value,'frame':None,'native_pts':None,'pts_seconds':None,'interval':None,'evidence':evidence,'uncertainty_reason':reason,'directly_viewed_scope':'See this record.review_coverage (full frame lists, sheet paths, native PNG list)',**kw}
data={
'00024':{
 'desc':'노란 택시 밴(시작부터 왼쪽 인접 차로), 별도 흰 세단(30.5–32.5초 앞오른쪽).',
 'tags':['full_2hz','dense_19_23_and_30_33','native_near'], 'native':[('native_near',621)], 'near':[600,639],
 'key':[('dense_19_23_and_30_33',600),('native_near',621),('dense_19_23_and_30_33',639),('dense_19_23_and_30_33',960)],
 'collision':'택시 오른쪽 후방과 자차 전방이 투영상 매우 가까우나 접촉 예상 면은 보닛 아래에 가려진다. 흰 세단은 오른쪽으로 화면 밖에 나가며 측면 접촉 영역을 확인할 수 없다. 어느 차량과 실제 접촉했는지 확정할 시각 증거가 없다.',
 'entry':'교차로에 보이는 점선은 근접 택시의 오른쪽에 남아 있다. 최초 바퀴가 자차 경계 또는 식별 가능한 연장선에 닿는 순간은 확보하지 못했다. 흰 세단도 실제 사고 상대와 진입 사건을 확정하지 못했다.',
 'side':'노란 택시 밴의 화면 LEFT 위치는 직접 관측했다. 실제 사고 상대의 진입 방향인지는 확인 불가.',
 'scene':'택시 오른쪽 전방에 교차로 노면이 보이며 오른쪽 횡단보도에는 보행자가 있다. 충돌 당시 이동·회피 가능 공간 0/1은 실제 접촉시점과 상대 미확정으로 판정하지 않는다.',
 'near_desc':'노란 택시 밴이 왼쪽 전방에서 보닛 부근과 투영상 가까워지는 구간. 충돌 존재나 최초 접촉의 상·하한을 의미하지 않는다.',
 'visible_side':'LEFT'},
'00025':{
 'desc':'왼쪽에서 자차 전방을 가로질러 우측 도로로 이동하는 검은 세단.',
 'tags':['dense_turn','native_entry_near'], 'native':[('native_entry_near',564)], 'near':[579,624],
 'key':[('dense_turn',525),('native_entry_near',564),('native_entry_near',600),('dense_turn',624),('dense_turn',642)],
 'collision':'검은 세단이 보닛 왼쪽 및 앞쪽에 매우 근접한다. 차체 하부와 자차 접촉 예상 면이 영상 하단에 가려져 실제 접촉·변형을 확인하지 못했다. 자차의 방향 변화와 근접만으로 충돌을 확정하지 않는다.',
 'entry':'교차로 이전 차선과 세단의 왼쪽 출발 위치는 보이지만, 회전 중 자차 차선 경계의 연장선과 세단의 첫 바퀴 접점이 일관되게 판독되지 않는다. 정확 프레임뿐 아니라 물리 진입을 묶는 검증된 양쪽 경계도 확보하지 못했다.',
 'side':'검은 세단이 화면 왼쪽에서 전방을 가로지르는 이동은 직접 보인다. 실제 사고 상대 확정과는 별도이다.',
 'scene':'근접 구간 오른쪽에는 연석·교통섬과 대기 보행자가 있고 왼쪽에는 세단이 있다. 실제 충돌시점이 미확정이므로 당시 도로 공간 0/1을 단정하지 않는다.',
 'near_desc':'세단이 자차 전방 왼쪽에서 보닛 바로 앞에 크게 보이고 자차 시야가 우측 도로를 향하는 구간. 실제 접촉시간 구간이 아니다.',
 'visible_side':'LEFT'},
'00026':{
 'desc':'화면 오른쪽에 보이는 파란 대형 트레일러(측면 prime 표시).',
 'tags':['dense_near'], 'native':[('dense_near',345)], 'near':[333,375],
 'key':[('dense_near',0),('dense_near',330),('dense_near',345),('dense_near',375),('dense_near',539)],
 'collision':'카메라가 위를 향하고 도로가 놓이는 화면 하단이 흐림 처리되어 트럭 바퀴·자차 외곽·접촉면을 볼 수 없다. 트럭 측면이 화면에 크게 나타나는 것만으로 충돌을 확정할 수 없다.',
 'entry':'자차 차선 경계와 트럭 바퀴가 보이지 않아 진입 프레임/시간 및 그 구간을 판독할 수 없다.',
 'side':'트레일러가 화면 RIGHT에 보이는 위치만 관측된다. 도로가 보이지 않아 실제 진입 방향 및 실제 사고 상대 여부는 확인 불가.',
 'scene':'주행 노면·차선과 장애물 배치를 판독할 수 없어 공간 0/1을 결정할 근거가 없다.',
 'near_desc':'트레일러 후면에서 측면이 화면 오른쪽을 크게 채우는 상대 시야 변화 구간. 거리·차선진입·충돌을 뜻하지 않는다.',
 'visible_side':'RIGHT'},
'00027':{
 'desc':'회전 후 같은 진행 경로 전방에서 추적되는 어두운 청색 승용 차량.',
 'tags':['dense_approach','native_nearest'], 'native':[('dense_approach',585)], 'near':[570,591],
 'key':[('dense_approach',0),('dense_approach',45),('dense_approach',570),('native_nearest',579),('dense_approach',585),('native_nearest',591)],
 'collision':'전방 차량의 후부가 커지며 자차 보닛과 투영상 가까워진 뒤 상대 위치가 안정된다. 양 차량의 범퍼 접촉면은 보닛·어둠에 가려져 접촉이나 변형을 직접 확인하지 못했다. 근접과 정지 자체는 충돌 증명이 아니다.',
 'entry':'클립 시작은 자차가 회전 중인 시야이다. 뒤에 추적하는 전방 차량을 첫 프레임의 작은 차량과 확정적으로 동일시하고, 그 차량이 당시 자차 차선에 이미 들어와 있었다고 결정할 수 없다. 경계를 가로지르는 최초 바퀴 시점도 관측되지 않았다.',
 'side':'근접 직전 차량은 이미 전방에 있다. 그 화면상 좌측 치우침을 LEFT 진입 방향으로 바꿀 근거가 없다.',
 'scene':'근접 시 왼쪽에는 반대방향 차량 불빛, 오른쪽에는 주차·정차 차량이 보인다. 실제 충돌시점과 차폭 여유가 확정되지 않아 공간 0/1은 알 수 없다.',
 'near_desc':'전방 차량이 보닛 부근까지 확대되고 상대 위치가 안정되는 구간. 범퍼 간 접촉의 시간 구간으로 사용하지 않는다.',
 'visible_side':None},
'00028':{
 'desc':'첫 원본 프레임부터 같은 차선 전방에 있는 검은 Nissan SUV.',
 'tags':['native_approach','overview_gap_fill'], 'native':[('native_approach',612)], 'near':[610,616],
 'key':[('native_approach',0),('native_approach',610),('native_approach',612),('native_approach',614),('native_approach',616)],
 'collision':'전방 SUV가 커지고 f611–616 부근에서 영상의 상하 자세 변화가 보인다. 그러나 자차와 SUV의 실제 범퍼 접촉면이 영상 하단에 가려져 있고 접촉·변형을 직접 확인하지 못했다. 급제동 또는 카메라 자세 변화와 충돌을 원본 시각 정보만으로 구별하지 못한다.',
 'entry':'관측한 검은 SUV는 첫 프레임부터 같은 차선 안에 있다. 이 차량이 실제 충돌 상대라고 별도로 확정되는 경우에 한하여 competition_entry는 f0/PTS0/0초이다. 물리적 최초 진입은 영상 시작 이전이며 시점 미상이다. 현재 실제 접촉 상대 자체는 미확정이므로 대회 정답으로 동결하지 않는다.',
 'side':'SUV는 시작부터 같은 차선 전방에 있어 진입 이전의 LEFT/RIGHT 기원을 볼 수 없다.',
 'scene':'전방은 SUV가 점유하고 오른쪽은 연석·보도이다. 왼쪽은 중앙선 너머 반대방향 도로이며 영상 일부에 통행 차량이 보인다. 실제 충돌시점과 회피 가능성은 검증되지 않아 0/1을 단정하지 않는다.',
 'near_desc':'전방 SUV가 매우 가까운 상태에서 상하 시야 변화가 집중되는 짧은 구간. 물리 접촉의 확정 구간은 아니다.',
 'visible_side':None},
'00029':{
 'desc':'18.5–21초 화면 오른쪽의 은색 소형 MPV; 별도로 31.5–35초 오른쪽에서 전방으로 오는 청회색 SUV.',
 'tags':['native_close_pass','dense_secondary_merge'], 'native':[('native_close_pass',604),('native_close_pass',564),('native_close_pass',579)], 'near':[591,616],
 'key':[('native_close_pass',564),('native_close_pass',579),('native_close_pass',604),('native_close_pass',616),('native_close_pass',630),('dense_secondary_merge',999)],
 'collision':'은색 MPV가 자차 오른쪽에 매우 가까이 있고 왼쪽 흰 세단도 병행한다. 접촉 예상 부위는 영상 하단·우측 바깥으로 가려지며, 직접적인 접촉·손상은 확인되지 않는다. 뒤의 청회색 SUV 합류에서도 자차와의 접촉 증거를 확보하지 못했다.',
 'entry':'은색 MPV가 오른쪽 차선 경계에 근접하지만, 경계의 점선 공백과 근접 시 바퀴·차체 하부의 가림으로 최초 바퀴가 경계에 닿는 프레임을 확정하지 못했다. 뒤의 청회색 SUV 이동은 별도 차량이며 이를 실제 사고 상대의 진입으로 대체하지 않는다.',
 'side':'은색 MPV와 후반 청회색 SUV가 화면 RIGHT에 있다는 사실은 직접 보인다. 실제 충돌 상대 여부는 별도로 확정되지 않았다.',
 'scene':'은색 MPV 근접 시 왼쪽 흰 세단과 앞쪽 차량들이 동시에 보인다. 시야가 보이는 노면을 실제 충돌 당시 안전하게 계속 진행·회피할 수 있는 공간으로 확정할 수 없다.',
 'near_desc':'은색 MPV가 오른쪽을 크게 채우고 왼쪽 흰 세단이 병행하는 가장 가까운 시각 구간. 실제 접촉 구간은 아니다.',
 'visible_side':'RIGHT'}
}
records=[]
for v,d in data.items():
 mapping=M[v]['mapping']; source=B/'acquisition'/f'{v}.mp4'; digest=hashlib.sha256(source.read_bytes()).hexdigest(); assert digest==M[v]['source_sha256']
 if v=='00024': overview=json.loads((O/v/'full_2hz/frames.json').read_text()); overview_images=list((O/v/'full_2hz').glob('sheet_*.jpg'))
 else:
  j=json.loads((shared/v/'manifest.json').read_text()); overview=[mapping[x['frame']] for x in j['frames']]; overview_images=list((shared/v).glob('sheet_*.jpg'))
 if v=='00028': overview+=json.loads((O/v/'overview_gap_fill/frames.json').read_text()); overview_images+=list((O/v/'overview_gap_fill').glob('sheet_*.jpg'))
 overview=sorted({x['frame_id']:x for x in overview}.values(),key=lambda x:x['frame_id'])
 details=[]; viewed={x['frame_id'] for x in overview}
 for tag in d['tags']:
  if tag in ('full_2hz','overview_gap_fill'):continue
  fs=json.loads((O/v/tag/'frames.json').read_text()); viewed.update(x['frame_id'] for x in fs)
  details.append({'tag':tag,'view_method':'all frames in all listed contact sheets visually inspected','frames':fs,'sheet_paths':[str(p) for p in sorted((O/v/tag).glob('sheet_*.jpg'))],'decode_native_pts_match':True})
 coverage={'overview_frames':overview,'overview_sheet_paths':[str(p) for p in sorted(overview_images)],'overview_max_gap_seconds':max(b['time_s']-a['time_s'] for a,b in zip(overview,overview[1:])), 'full_time_axis_covered':True,'dense_inspections':details,'standalone_native_pngs':[ev(v,t,f,True) for t,f in d['native']], 'unique_directly_viewed_frame_count':len(viewed),'all_directly_viewed_frame_ids':sorted(viewed),'audio_reviewed':False,'all_original_frames_watched':False}
 assert coverage['overview_max_gap_seconds']<=0.50000001
 evid=[ev(v,t,f,(t,f) in d['native']) for t,f in d['key']]
 fields={'collision':item('unknown',None,d['collision'],evid),'entry':item('unknown',None,d['entry'],evid),'direction':item('unknown',None,d['side'],evid),'space':item('unknown',None,d['scene'],evid)}
 track={'description':item('observed',d['desc'],'관측된 차량 기술이며 실제 충돌 상대의 확정이 아니다.',evid), 'actual_accident_counterparty':item('unknown',None,d['collision'],evid), 'screen_side':item('observed' if d['visible_side'] else 'unknown',d['visible_side'],d['side'],evid),'physical_entry':item('unknown',None,d['entry'],evid)}
 if v=='00028':
  p=point(v,0);track['conditional_competition_entry']=item('observed',0,'이 SUV가 실제 충돌 상대라고 별도 확정될 때만 대회 convention으로 적용한다. 물리 진입을 0초로 주장하지 않는다.',[ev(v,'native_approach',0)],frame=p['frame_id'],native_pts=p['native_pts'],pts_seconds=p['time_s'],time_base=p['time_base'],condition='only if this observed SUV is established as actual collision counterparty')
  track['physical_entry']['interval']={'upper_bound_exclusive':point(v,0),'lower_bound':None};track['physical_entry']['value']='before_clip_start_time_unknown'
 rec={'id':v,'type':'ai_expert_independent_review','reviewer_id':'reviewer_a','human_review':False,'model_predictions_seen':False,'other_reviews_seen':False,'model_inference_run':False,'source':{'video_path':str(source),'video_sha256':digest,'pts_path':str(B/'acquisition'/f'{v}.pts.json'),'frame_count':len(mapping),'first':mapping[0],'last':mapping[-1]},'review_coverage':coverage,'physical_contact_surface_directly_visible':item('unknown',None,'검토한 근접 구간에서 실제로 맞닿는 양 차량의 접촉면을 동시에 판독하지 못했다. 이는 충돌이 없었다는 뜻이 아니다.',evid),'visual_incident_candidate':item('inferred_interval',d['near_desc'],'시각상 근접/상대시야 변화로 선택한 재검토 구간이며 물리적 충돌 존재나 최초 접촉의 상하한을 주장하지 않는다.',evid,interval=interval(v,*d['near']),is_collision_time_interval=False),'observed_vehicle_track':track,'competition_targets':fields,'suitable_for_hard_ground_truth':False,'review_completion':'completed_with_explicit_abstentions','limitations':['전 시간축을 0.5초 이하 간격으로 직접 판독했으나 전체 원본의 모든 프레임을 연속 재생한 검수는 아니다.','무음 시각 검수이며 센서·외부 사고 사실을 사용하지 않았다.','unknown은 미발생 또는 negative 라벨이 아니다.','근접·화면 흔들림만으로 실제 최초 접촉을 확정하지 않았다.']}
 (O/f'{v}.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');records.append(rec)
result={'type':'ai_expert_independent_review','reviewer_id':'reviewer_a','human_review':False,'model_predictions_seen':False,'other_reviews_seen':False,'model_inference_run':False,'allowed_sources_only':True,'definition':{'collision':'자차와 실제 동일 상대의 최초 실제 접촉','entry':'그 상대 첫 바퀴가 자차 차선 경계(판별 가능한 교차로 연장선 포함)에 닿는 때','entry_start_convention':'시작부터 이미 진입이면 competition_entry는 첫 원본 프레임; physical entry는 시작 이전 시점 미상','direction':'화면 LEFT/RIGHT','space':'실제 충돌 당시 계속 진행/회피 가능한 도로 공간 0/1'},'status_semantics':{'observed':'직접 시각 증거에서 읽을 수 있는 사실. 조건부 track 관찰을 실제 사고 정답으로 승격하지 않는다.','inferred_interval':'기록된 근거에서 선택한 추정 시간 범위. 본 기록의 visual_incident_candidate는 물리 충돌 구간이 아니다.','unknown':'원본 시각자료에서 확정할 수 없음; negative 라벨 아님'},'records':records}
(O/'records.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print([(x['id'],x['review_coverage']['unique_directly_viewed_frame_count'],x['review_coverage']['overview_max_gap_seconds']) for x in records])
