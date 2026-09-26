import json
from pathlib import Path
P=Path('artifacts/stage1_tpo_center70_20260918')
s=json.loads((P/'summary.json').read_text());rows=json.loads((P/'predictions.json').read_text());changes=json.loads((P/'decision_changes.json').read_text());h=s['splits']['holdout']['primary_all'];b,c=h['baseline'],h['candidate'];t=s['timing']
assert all(abs(r['candidate']-.5*(sum(r['candidate_frame_scores'][:12])/12+sum(r['candidate_frame_scores'][12:])/12))<1e-12 for r in rows)
assert all(len(r['full_frame_scores'])==12 and len(r['candidate_frame_scores'])==24 for r in rows)
assert s['files_unchanged']
lines=['# TPO 전체 화면 + 중앙 70%: 단일 고정 후보 실험','',
'2026-09-18. 기존 16원천/384프레임의 동일 입력에서 요청한 후보 하나만 시험했다. 비율·크롭·임계값 탐색, 재학습, 제출 설정 변경은 하지 않았다.','',
'## 결과','',
'기존 최종 확인용 8원천의 기본·JPEG75·블러σ1 조건 합산 결과. 클래스별 24건은 8원천×3조건의 반복 기록이며 독립 표본 24개가 아니다. 이 자료는 이전 결과가 이미 노출되어 이번 실험의 새 독립 holdout은 아니다.','',
'| 방식 | Macro-F1 | 원본 오탐 /24 | 재촬영 누락 /24 |','|---|---:|---:|---:|',
f"| 전체 화면 TPO /0.5 | {b['macro_f1']:.4f} | {b['fp']} | {b['fn']} |",f"| 전체+중앙70% 50:50 /0.5 | {c['macro_f1']:.4f} | {c['fp']} | {c['fn']} |",'',
f"Macro-F1 변화 {c['macro_f1']-b['macro_f1']:+.4f}; 원본 오탐 변화 {c['fp']-b['fp']:+d}건; 재촬영 누락 변화 {c['fn']-b['fn']:+d}건.",
f"원본에서 새 오탐 {h['original_new_fp']}건/해소 {h['original_fixed_fp']}건, 재촬영에서 새 누락 {h['recapture_new_fn']}건/해소 {h['recapture_fixed_fn']}건.",'',
'### 조건별 결과','',
'각 조건은 클래스별 8건. 기본 입력은 앞 실험에서 구성한 테두리 없는 640×360/JPEG95 공통처리 영상 프레임이다.','',
'| 조건 | 기준 F1 | 후보 F1 | 기준 오탐/누락 | 후보 오탐/누락 |','|---|---:|---:|---:|---:|']
names={'full_frame':'기본','jpeg_75':'JPEG75','blur_1':'블러σ1','jpeg_50':'JPEG50','blur_2':'블러σ2','resize_050':'1/2 축소'}
for mode,name in names.items():
 x=s['splits']['holdout'][mode];a,d=x['baseline'],x['candidate'];lines.append(f"| {name} | {a['macro_f1']:.4f} | {d['macro_f1']:.4f} | {a['fp']}/{a['fn']} | {d['fp']}/{d['fn']} |")
lines+=['','개발용 결과는 참고로 별도 표시한다. 이번 실험에서는 개발용 결과로 후보를 조정하지 않았다.','',
'| 개발용 주요 3조건 | Macro-F1 | 원본 오탐 /24 | 재촬영 누락 /24 |','|---|---:|---:|---:|']
for k,name in [('baseline','전체 화면'),('candidate','전체+중앙70%')]:
 v=s['splits']['development']['primary_all'][k];lines.append(f"| {name} | {v['macro_f1']:.4f} | {v['fp']} | {v['fn']} |")
lines+=['','## 실행 시간','',
'Apple M4 Max, CPU 추론 2스레드. 한 번 적재하고 워밍업한 동일 모델에서 192개 영상·조건 기록을 각각 두 방식으로 실제 재추론했다. 기준 결과를 캐시로 대신하지 않았다. 매 기록마다 실행 순서를 교대했다. 후보는 전체12+크롭12를 batch_size8로 처리한다.','',
'| 방식 | 평균/12프레임 묶음 | 중앙값 | 192기록 합계 |','|---|---:|---:|---:|']
for method,name in [('baseline','전체 화면'),('candidate','전체+중앙70%')]:
 v=t[method];lines.append(f"| {name} | {v['mean_seconds_per_12_frame_record']*1000:.1f}ms | {v['median_seconds']*1000:.1f}ms | {v['total_seconds']:.2f}s |")
lines +=['',f"후보/기준 총 실행 시간 비율: {t['total_ratio']:.2f}배. 공통 모델 적재는 별도 {s['model_load_seconds']:.2f}초.",
'시간에는 TPO 전처리·forward·평균 및 후보 크롭이 포함된다. 공통 파일 읽기·조건 변환·모델 적재·MP4 디코딩은 제외한다. 실제 대회 전체 실행 시간의 배율은 이 측정으로 확정할 수 없다.','',
'## 고정 설정과 검증','',
'- 각 프레임에서 가로·세로 각각70%의 중앙 영역(면적49%)을 자른 뒤 기존 TPO의224×224 전처리를 적용했다. 기본640×360→448×252, 축소320×180→224×126. 선행 확대 보간 없음.',
'- 기준: 동일12프레임 전체 점수의 평균. 후보: 전체12프레임 평균과 크롭12프레임 평균을50:50 결합. 두 방식 모두 RERECORDED 판정은 점수≥0.5.',
'- 6조건×16원천×2클래스=192기록. 기존 기본/변환 입력·프레임 순서·모델을 동일하게 유지했다.',
f"- 기존 TPO 기준값과 최대 차이 {s['max_previous_baseline_score_difference']:.3g}; 후보에서 재계산한 전체 프레임 점수와 기준의 최대 차이 {s['max_repeated_full_frame_difference']:.3g}. 전후 파일 해시 불변 확인.",
'- 중앙 크롭 좌표·배열 일치 self-check, 기록 유일성/개수, 프레임별 확률의 결합식 및 점수 범위를 확인했다.',
'- 작은 거리 영상 세트·한 재촬영 장비·반복 사용된 평가 자료라는 기존 한계가 유지된다. 크롭은 내용과 시야도 바꾸므로 변화 원인을 모아레 확대 하나로 단정할 수 없다.','',
'## 판정이 바뀐 사례','',
'아래는 기존 최종 확인용의 주요3조건에서 판정이 바뀐 모든 사례다. 원천 번호는 REDS train ID이다.','',
'| 원천 | 정답 | 조건 | 전체 점수 | 중앙 점수 | 결합 점수 | 변화 |','|---|---|---|---:|---:|---:|---|']
for r in changes:
 if r['split']=='holdout' and r['mode'] in ['full_frame','jpeg_75','blur_1']:
  good=(r['candidate']>=.5)==(r['label']=='recapture');lines.append(f"| {r['source']:03d} | {r['label']} | {names[r['mode']]} | {r['baseline']:.4f} | {r['center']:.4f} | {r['candidate']:.4f} | {'정정' if good else '새 오류'} |")
lines+=['','## 재현 파일','',
'- 실행: `artifacts/mac_experiments/scipy_compat/.venv/bin/python scripts/data/evaluate_stage1_tpo_center70.py` (기존 출력 폴더가 있으면 덮어쓰지 않고 중단).',
'- `artifacts/stage1_tpo_center70_20260918/`: protocol.json, predictions.json, summary.json, decision_changes.json, environment.json.',
'- 기존 자료·전처리의 세부 근거: `docs/stage1-road-recapture-comparison-20260918.md`.','']
Path('docs/stage1-tpo-center70-comparison-20260918.md').write_text('\n'.join(lines))
print(json.dumps({'holdout_primary':h,'development_primary':s['splits']['development']['primary_all'],'timing':t},indent=2))
