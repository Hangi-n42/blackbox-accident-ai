import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
base=Path(__file__).resolve().parent
root=base.parents[1]
checks={d['clip_id']:d for d in json.loads((base/'source_checks.json').read_text())}
PROV={'type':'ai_expert_independent_review','reviewer_id':'reviewer_b','human_review':False,'model_predictions_seen':False,'other_reviews_seen':False,'prior_annotations_seen':False,'provider_event_labels_seen':False,'model_inference_executed':False,'external_sources_used':False,'visual_inspection_tool':'view_image','generated_at_utc':datetime.now(timezone.utc).isoformat()}
D={
'00024':{
 'candidate':'노란색 박스형 택시. f0의 왼쪽 인접 차량을 f570~660까지 차체 형상·색·창문·후미등의 연속성으로 추적했다. 첫 화면 전방의 청색 SUV와는 다른 차량이다.',
 'track_status':'observed','track':[0,660],
 'reason':'f600~627에서 택시 후면/우측 차체와 자차 후드가 화면상 가까워지나 실제 양 차량 접촉면은 후드 아래에 가려진다. 택시 변형·접촉 지점은 보이지 않는다. 그 뒤 택시는 계속 주행한다. f945~990의 흰색 세단 우측 근접도 별도로 확인했으나 접촉면은 보이지 않는다. 이 두 근접을 실제 사고 상대 확정으로 사용하지 않았다.',
 'focus':[600,627], 'secondary_focus':[945,990],
 'entry_reason':'f570,600,615에서 택시 우측 바퀴는 화면에 보이는 점선 좌측에 남아 있다. 교차로에서 자차 방향이 바뀌어 상대 바퀴가 자차 차로 경계에 최초로 닿는 순간을 특정할 수 없다. 화면상 접근과 차로 진입을 구분했다.',
 'screen_side':'LEFT','side_reason':'주요 추적 택시는 화면 왼쪽에서 보인다. 그러나 자차 차로 진입 자체와 사고 상대가 확정되지 않아 대회 LEFT 정답으로 승격하지 않았다.',
 'unseen':['후드 아래 자차 전방 범퍼와 택시 접촉면','가림 구간에서의 실제 두 차량 간 거리','오디오 및 측후방 영상','교차로 통과 중 바퀴와 자차 차로 경계의 최초 접점'],
 'evidence':['00024_f0000.png','00024_0570_0660_s3/f0570.png','00024_0570_0660_s3/f0600.png','00024_0570_0660_s3/f0615.png','00024_0600_0627_s1/sheet_01.jpg','00024_0945_0990_s3/sheet_00.jpg'],
},
'00025':{
 'candidate':'검은색 세단. 첫 화면 왼쪽 인접 차로의 세단을 f510~645까지 창문 윤곽·차체·후미등을 통해 추적했다. f597~620에서 자차 전방 왼쪽 차체가 근접하고 f624 이후 앞을 가로질러 진행한다.',
 'track_status':'observed','track':[0,645],
 'reason':'f597~620에서 세단 우측 전면/측면이 후드 왼쪽 아래와 겹쳐 보이고 자차 시야가 우측 보도·교통섬으로 돌아간다. 실제 범퍼/차체 접촉점은 화면 하단에 가려진다. 보행자 앞 정지와 급격한 시야 변화만으로 접촉을 확정하지 않았다. 세단은 이후 전방으로 떠난다.',
 'focus':[597,620],
 'entry_reason':'f510~549 연속 원본에서 우측 바퀴와 왼쪽 차로 경계 근처를 확인했다. 교차로 진입 이후 차로 표시가 끊기고 자차 회전과 와이퍼/후드 가림이 겹친다. 첫 바퀴 접점을 확인할 수 없고 보수적인 물리 진입 구간도 확정하지 않았다.',
 'screen_side':'LEFT','side_reason':'추적 세단은 화면 왼쪽에서 자차 앞쪽으로 이동한다. 조건부 화면상 출발 방향만 LEFT이며 실제 사고 상대의 대회 방향은 미상이다.',
 'unseen':['후드 아래 세단 바퀴 및 자차 전방/좌측 접촉면','끊긴 교차로 차선 연장선의 정확한 위치','자차와 세단의 실제 거리','오디오 및 측후방 영상'],
 'evidence':['00025_f0000.png','00025_0510_0549_s1/f0510.png','00025_0510_0549_s1/f0529.png','00025_0510_0549_s1/f0535.png','00025_0540_0645_s3/f0570.png','00025_0540_0645_s3/f0609.png','00025_0597_0620_s1/sheet_01.jpg'],
},
'00026':{
 'candidate':'파란색 Prime 표기가 있는 대형 트레일러. f300~465에서 후면에서 좌측 측면으로 이어지는 같은 트레일러를 추적할 수 있다. 앞서 보이는 흰색 트럭과는 구분한다.',
 'track_status':'observed','track':[300,539],
 'reason':'처음부터 카메라가 하늘을 향하고 하단 넓은 영역은 블러 처리되어 있다. f340 이후 파란 트레일러 측면이 화면 오른쪽을 크게 차지하지만, 도로·바퀴·자차 차체 접점은 보이지 않는다. 트럭 측면의 확대와 카메라 움직임은 접촉의 직접 증거가 아니다.',
 'focus':[360,375],
 'entry_reason':'전체 0~539 표본에서 상대 바퀴와 자차 차로 경계를 동시에 판독할 수 없다. 측면이 오른쪽에서 나타난 사실을 바퀴의 차선 최초 접촉으로 대체할 수 없다.',
 'screen_side':'RIGHT','side_reason':'파란 트레일러는 화면 오른쪽에 보인다. 차로 경계가 보이지 않아 진입 방향 RIGHT로 확정하지 않았다.',
 'unseen':['도로와 자차 차로 경계','트레일러 바퀴','실제 접촉점','진행 및 회피 공간','하단 블러 내부 내용','오디오 및 측후방 영상'],
 'evidence':['00026_f0000.png','00026_0300_0465_s5/sheet_00.jpg','00026_0300_0465_s5/sheet_02.jpg','00026_0360_0375_s1/sheet_00.jpg'],
},
'00027':{
 'candidate':'전방의 짙은 청색 승용차. f0 오른쪽에 보이는 차량과 f60 이후 전방 차량의 연결은 어둠·차량 중첩 때문에 확정하지 않았다. f60~1199의 전방 차체와 후미등은 일관되게 추적할 수 있다.',
 'track_status':'observed','track':[60,1199],
 'reason':'f573~596에서 앞차가 후드 가까이 확대되고 정지한다. f581~589 주변에도 실제 자차 범퍼와 앞차의 접촉점은 보이지 않는다. 이후 사람이 차량 부근으로 나오는 모습은 확인되지만 접촉의 직접 증거로 사용하지 않았다.',
 'focus':[573,596],
 'entry_reason':'첫 화면에서 자차는 회전 중이고 대상 동일성/차로 경계가 불분명하다. f60부터 전방 같은 통행 경로에 있는 사실은 확인되지만 최초 바퀴 진입은 보이지 않는다. 첫 이미지부터 차로 안이라는 조건이 입증되지 않아 competition entry=0을 부여하지 않았다.',
 'screen_side':None,'side_reason':'정밀검수 시점에는 이미 전방이다. 화면 좌우에서의 최초 진입은 관찰되지 않는다.',
 'unseen':['처음 f0~60 대상 차량 동일성의 확실한 연결','최초 진입 당시 차로 경계 및 바퀴','후드 아래 실제 접촉면','야간 측방 빈 공간의 전체 폭','오디오 및 측후방 영상'],
 'evidence':['00027_f0000.png','00027_0000_0120_s10/sheet_00.jpg','00027_0000_0120_s10/sheet_01.jpg','00027_0540_0615_s3/f0582.png','00027_0573_0596_s1/sheet_01.jpg'],
},
'00028':{
 'candidate':'첫 프레임부터 전방에 있는 짙은색 Nissan SUV. f0~1224 전체 시간축에서 같은 후면 차체·엠블럼·등화의 연속성을 확인했다.',
 'track_status':'observed','track':[0,1224],
 'reason':'f585~613에서 앞차가 가까워지고 f614~621에서 시야와 앞차 위치가 변한다. 그러나 실제 자차 범퍼/앞차 접촉면은 하단 가림 아래에 있다. 근접·등화 변화·시야 흔들림만으로 실제 접촉 또는 최초 접촉 프레임을 확정하지 않았다.',
 'focus':[606,621],
 'entry_reason':'동일 추적 SUV가 f0에서 노란 중앙선 오른쪽과 우측 연석 사이의 자차 통행 차로 안에 이미 있다. 이 차량이 실제 사고 상대라고 추가 확정될 경우 대회 규칙값은 첫 원본번호 0이다. 물리적 진입은 영상 시작 전 미상이며 0초 진입을 뜻하지 않는다.',
 'screen_side':None,'side_reason':'첫 화면부터 이미 전방 차로 안이어서 LEFT/RIGHT 물리 진입 방향은 관찰되지 않는다.',
 'unseen':['영상 시작 전 실제 진입 시각/방향','하단 자차 전방 범퍼와 접촉점','실제 접촉 순간','접촉 순간을 전제로 하는 회피 공간','오디오 및 측후방 영상'],
 'evidence':['00028_f0000.png','00028_0585_0635_s2/f0611.png','00028_0606_0621_s1/sheet_00.jpg','00028_0606_0621_s1/sheet_01.jpg'],
},
'00029':{
 'candidate':'우측 차로에서 접근하는 은색 소형 해치백/MPV. f480~645에서 후면 램프·검은 뒤창·차체를 연속 추적했다. 시작 프레임의 회색 세단 및 좌측의 흰색 세단과 구분했다.',
 'track_status':'observed','track':[480,645],
 'reason':'f597~620에서 은색 차량 좌측면이 화면 오른쪽을 크게 차지하고 왼쪽에 흰색 세단이 병행한다. 자차 측면과 은색 차량의 실제 접촉점은 우측/하단 시야 밖이다. 차체 변형이나 두 차체 접촉을 직접 볼 수 없고 이후 두 차량은 계속 주행한다.',
 'focus':[597,620],
 'entry_reason':'f570에서 은색 차량의 좌측 바퀴는 우측 인접 차로 쪽에 남아 있다. f585에서는 같은 바퀴가 보이는 점선의 연장선보다 자차 차로 쪽에 들어와 있다. 그 사이 점선 공백·차량 가림·자차 방향 변화 때문에 최초 접점을 한 프레임으로 정할 수 없어 조건부 진입 구간을 f570~585로 보존했다.',
 'screen_side':'RIGHT','side_reason':'추적 은색 차량은 화면 오른쪽 인접 차로에서 접근한다. 조건부 candidate entry 방향은 RIGHT이며 실제 사고 상대의 대회 정답은 접촉 미확정으로 보류한다.',
 'unseen':['우측/하단 시야 밖 자차 측면과 상대 차체 접점','점선 공백 및 상대 하부 가림의 정확한 최초 바퀴 접점','실제 접촉 시각','자차 좌우 전체 측방 여유 폭','오디오 및 측후방 영상'],
 'evidence':['00029_f0000.png','00029_0555_0645_s3/f0555.png','00029_0555_0645_s3/f0570.png','00029_0563_0578_s1/f0578.png','00029_0555_0645_s3/f0585.png','00029_0555_0645_s3/f0603.png','00029_0597_0620_s1/sheet_01.jpg'],
}}
records=[]
for clip,d in D.items():
 mapping=json.loads((root/'acquisition'/f'{clip}.pts.json').read_text())['mapping']
 def at(f): return dict(mapping[f])
 def interval(a,b):return {'lower':at(a),'upper':at(b),'inclusive':True}
 overview=json.loads((base.parent/'blind_packet'/clip/'manifest.json').read_text())
 overview_ids=[x['frame'] for x in overview['frames']]
 native=[];native_ids=set()
 for p in sorted(base.glob(f'{clip}_*_s*/manifest.json')):
  rows=json.loads(p.read_text());ids=[r['frame_id'] for r in rows];native_ids.update(ids)
  native.append({'manifest':str(p),'frames':ids,'sheets_actually_viewed':[str(q) for q in sorted(p.parent.glob('sheet_*.jpg'))]})
 evidence=[]
 for rel in d['evidence']:
  path=base/rel;assert path.exists(); ev={'image_path':str(path),'reason':'직접 view_image 판독한 근거 이미지'}
  if path.stem.startswith('f') and path.stem[1:].isdigit(): ev.update(at(int(path.stem[1:])))
  elif rel.endswith('_f0000.png'): ev.update(at(0))
  evidence.append(ev)
 cond_entry={'status':'unknown','exact':None,'interval':None,'reason':d['entry_reason']}
 if clip=='00028':cond_entry={'status':'observed','rule_based_frame':at(0),'physical_entry':{'status':'unknown','exact':None,'interval':{'lower':None,'upper_exclusive':at(0)},'reason':'시작 이전 미상'},'reason':d['entry_reason']}
 if clip=='00029':cond_entry={'status':'inferred_interval','exact':None,'interval':interval(570,585),'reason':d['entry_reason']}
 unknown=lambda why:{'status':'unknown','value':None,'exact':None,'interval':None,'reason':why}
 r={'clip_id':clip,'provenance':dict(PROV),'source':{'mp4':str(root/'acquisition'/f'{clip}.mp4'),'pts_json':str(root/'acquisition'/f'{clip}.pts.json'),'frame_id_convention':'0-based original decode order','sha256':checks[clip]['sha256'],'source_and_pts_verified':checks[clip]['sha_matches'] and checks[clip]['pts_matches'],'frame_count':len(mapping),'time_base':mapping[0]['time_base'],'last_frame':at(len(mapping)-1)},
 'direct_observation_scope':{'overview':'전체 시간축 약 2Hz 원본 이미지 시트를 실제 view_image 판독. 영상 연속 재생/오디오 청취 아님. 선택 프레임 사이 모든 원본 프레임을 판독했다는 뜻이 아님.','overview_frame_ids':overview_ids,'overview_sheets_actually_viewed':[str(p) for p in sorted((base.parent/'blind_packet'/clip).glob('sheet_*.jpg'))],'native_detail_sets':native,'unique_reviewed_frames':len(set(overview_ids)|native_ids|{0}),'unseen_information':d['unseen']},
 'candidate_identity':{'status':'observed','description':d['candidate'],'consistent_tracking_interval':interval(*d['track']),'actual_accident_counterpart_confirmed':False,'actual_accident_counterpart_status':'unknown'},
 'ego_collision_occurred':unknown(d['reason']),
 'first_ego_contact':unknown('실제 자차 접촉이 확정되지 않았다. 아래 정밀검수/근접 구간을 충돌 구간으로 사용하면 안 된다.'),
 'reviewed_proximity_focus':{'status':'observed','interval':interval(*d['focus']),'is_contact_interval':False,'reason':'전체 시각 검수에서 자차와 추적 차량이 가까워지는 것으로 보여 원본 연속 프레임을 직접 확인한 범위. 실제 거리 측정이나 접촉 확정이 아니다.'},
 'conditional_candidate_entry':cond_entry,
 'conditional_candidate_screen_side':{'status':'observed' if d['screen_side'] else 'unknown','value':d['screen_side'],'reason':d['side_reason'],'is_verified_accident_entry_side':False},
 'competition_entry':unknown('같은 실제 사고 상대 확정이 선행되지 않았다. conditional_candidate_entry만 별도 참조.'),
 'competition_side':unknown('같은 실제 사고 상대 확정이 선행되지 않았다. 화면상 접근 위치를 대회 정답으로 자동 변환하지 않았다.'),
 'space_at_actual_contact':unknown('실제 접촉 시각이 확정되지 않아 그때의 진행/회피 공간 0/1도 정할 수 없다. 보이는 전방 빈 공간만으로 측후방 또는 전체 회피 여유를 보장하지 않는다.'),
 'evidence':evidence,
 'admissibility':{'ready_as_contact_ground_truth':False,'reason':'독립 AI 시각 검수이며 실제 접촉 및 같은 사고 상대가 확정되지 않았다. 원본/동기화 추가 증거로 판단해야 한다.'}}
 if 'secondary_focus'in d:r['additional_review_focus']={'status':'observed','interval':interval(*d['secondary_focus']),'is_contact_interval':False,'reason':'흰색 세단 우측 근접의 별도 검수'}
 records.append(r)
(base/'records.json').write_text(json.dumps({'schema_version':'1.0','provenance':PROV,'important':'unknown은 비충돌 판정이 아니다. conditional_candidate_*는 실제 사고 상대의 확정 대회 정답이 아니다. proximity focus는 충돌 시간 구간이 아니다.','records':records},ensure_ascii=False,indent=2)+'\n')
lines=['# Stage2 독립 AI 전문가 시각 검수 B','', '검수 대상은 00024~00029 원본 MP4와 그 PTS 매핑이다. 다른 검수자 기록, 기존 AI screening/REPORT, 인간 초안, 모델 예측/점수/후보/이벤트 라벨은 읽지 않았고 모델 추론도 실행하지 않았다. `human_review=false`, `type=ai_expert_independent_review`이다.','', '전체 2Hz 공유시트 444개 표본을 실제 이미지 도구로 판독했고, 각 영상의 근접·진입 검토 구간은 추가 원본 프레임으로 판독했다. 전체 원본을 연속 재생하거나 오디오를 청취하지 않았다. 모든 원본의 SHA256 및 순차 디코드 PTS 목록이 제공된 PTS JSON과 일치한다. 프레임 번호는 0-based이며 00028은 time_base=1/19584, 나머지는 1/15360이다.','', '**결론: 여섯 영상 모두 이 시각 증거만으로 실제 자차 접촉을 확정하지 않았다. 이는 비충돌 판정이 아니다.** 따라서 최초 접촉 프레임과 실제 접촉 당시 공간 0/1, 실제 사고 상대를 전제로 하는 확정 대회 정답은 미상으로 보존한다. 아래 근접 검수 범위는 접촉의 불확실성 구간으로 사용할 수 없다.','', '| 원본 | 추적 대상 | 조건부 진입 | 실제 접촉 / space |','|---|---|---|---|']
for r in records:
 c=r['clip_id']; e='미상'
 if c=='00028':e='규칙상 f0 / 물리 진입은 시작 전 미상'
 if c=='00029':e='추론 구간 f570~585, RIGHT (상대 확정 전 조건부)'
 lines.append(f"| {c} | {D[c]['candidate'].split('.')[0]} | {e} | 미상 / 미상 |")
for r in records:
 c=r['clip_id'];d=D[c];focus=r['reviewed_proximity_focus']['interval'];a=focus['lower'];b=focus['upper']
 lines += ['',f'## {c}','',f"추적 차량: {d['candidate']}",'',f"사고 상대 확정 여부: **미확정**. {d['reason']}",'',f"원본 연속 정밀검수 중심 범위: f{a['frame_id']}~f{b['frame_id']}, native PTS {a['native_pts']}~{b['native_pts']}, {a['time_s']:.6f}~{b['time_s']:.6f}초. 이 범위는 실제 충돌 시각의 bounds가 아니다.",'',f"진입: {d['entry_reason']}",'',f"방향: {d['side_reason']}",'','공간: 실제 접촉 시각이 미확정이므로 접촉 당시 진행·회피 공간 0/1은 미상이다.','',f"직접 본 범위: 전체 약 2Hz 시트와 다음 원본 프레임 세트. 합집합 {r['direct_observation_scope']['unique_reviewed_frames']}프레임."]
 for s in r['direct_observation_scope']['native_detail_sets']:
  ids=s['frames'];step=ids[1]-ids[0] if len(ids)>1 else 1
  lines.append(f"- f{ids[0]}~f{ids[-1]}, step={step}: [{Path(s['manifest']).parent.name}]({s['manifest']})")
 lines += ['', '볼 수 없던 정보: '+', '.join(d['unseen'])+'.','', '대표 근거 이미지:']
 for ev in r['evidence']:
  label=Path(ev['image_path']).name
  if 'frame_id'in ev:label+=f" (f{ev['frame_id']}, PTS {ev['native_pts']}, {ev['time_s']:.6f}s)"
  lines.append(f"- [{label}]({ev['image_path']})")
lines += ['','## 사용 제한','', '여기에 기록한 조건부 차량 추적·진입 관찰을 충돌 확정 GT로 사용해서는 안 된다. 00028의 f0 규칙값은 같은 앞차가 실제 사고 상대라고 별도 확정될 때에만 적용된다. 00029의 f570~585 구간은 차선/바퀴 기하에 대한 보수적 추론이며 정확한 한 프레임이 아니다. 00024~00027의 진입은 미상이다. 원본 영상의 비가림 접촉 장면, 추가 측면 영상 또는 동기화된 접촉 증거가 없으면 정확도를 숫자로 강제하지 않는다.','', '모든 항목의 status, native PTS, 관찰 범위, 근거 경로는 [records.json]('+str(base/'records.json')+')에 구조화했다.']
(base/'review.md').write_text('\n'.join(lines)+'\n')
print([(r['clip_id'],r['direct_observation_scope']['unique_reviewed_frames'])for r in records])
print(base/'records.json');print(base/'review.md')
