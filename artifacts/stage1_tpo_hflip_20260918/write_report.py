import json,subprocess
from pathlib import Path
P=Path('artifacts/stage1_tpo_hflip_20260918');dev=json.loads((P/'development/summary.json').read_text());passed=dev['development_gate_pass'];splits=['development','holdout'] if passed else ['development']
if passed:assert (P/'holdout/summary.json').exists()
else:assert not (P/'holdout').exists()
lines=['# TPO 좌우 반전 TTA 고정 후보 실험','',
'2026-09-18. 원래 화면12프레임과 좌우 반전12프레임의 TPO 확률을50:50 평균하고 임계값0.5로 비교했다. 재학습·비율/임계값 탐색·모델 교체 없음.','',
'## 진행 기준','',
'추론 전에 기본·JPEG75·블러σ1 조정용 합산 Macro-F1이 기준보다 엄격히 높을 때만 기존 확인용 평가를 진행하도록 고정했다. 그 외에는 중단한다. 블러/JPEG 등은 기존 평가 조건이며 이번 TTA에 추가한 증강이 아니다. 좌우 반전은 픽셀 열 순서만 뒤집으며, 기존 TPO 전처리 이외의 보간·압축은 없다.','',
f"조정용 통과 여부: **{'통과 — 고정 설정으로 확인용 진행' if passed else '실패 — 확인용을 실행하지 않고 후보 폐기'}**.",
'', '클래스별24건은8원천×주요3조건의 반복 기록이다. 독립 장면24개가 아니다. 같은12프레임을 영상 단위로 집계하며 전체16원천 자료의 기존 분할을 유지했다.','']
for split in splits:
 s=json.loads((P/split/'summary.json').read_text());rr=json.loads((P/split/'predictions.json').read_text());assert len(rr)==96 and s['files_unchanged']
 for r in rr:
  assert len(r['full_frame_scores'])==12 and len(r['candidate_frame_scores'])==24
  assert abs(r['candidate']-.5*(sum(r['candidate_frame_scores'][:12])/12+sum(r['candidate_frame_scores'][12:])/12))<1e-12
 r=s['splits'][split]['primary_all'];b,c=r['baseline'],r['candidate'];t=s['timing'];title='조정용' if split=='development' else '기존 확인용'
 lines += [f'## {title} 결과','', '| 방식 | Macro-F1 | 원본 오탐 /24 | 재촬영 누락 /24 |','|---|---:|---:|---:|',
 f"| 원래 화면 | {b['macro_f1']:.4f} | {b['fp']} | {b['fn']} |",f"| 원래+반전50:50 | {c['macro_f1']:.4f} | {c['fp']} | {c['fn']} |",'',
 f"Macro-F1 변화 {c['macro_f1']-b['macro_f1']:+.4f}. 원본 새 오탐 {r['original_new_fp']}건/해소 {r['original_fixed_fp']}건, 재촬영 새 누락 {r['recapture_new_fn']}건/해소 {r['recapture_fixed_fn']}건.",'',
 '| 조건 | 기준 F1 | 후보 F1 | 기준 오탐/누락 | 후보 오탐/누락 |','|---|---:|---:|---:|---:|']
 for m in ['full_frame','jpeg_75','blur_1','jpeg_50','blur_2','resize_050']:
  a=s['splits'][split][m]['baseline'];d=s['splits'][split][m]['candidate'];lines.append(f"| {m} | {a['macro_f1']:.4f} | {d['macro_f1']:.4f} | {a['fp']}/{a['fn']} | {d['fp']}/{d['fn']} |")
 lines+=['','各条件の分母はクラスごと8。'.replace('各条件の分母はクラスごと8。','각 조건의 분모는 클래스별8이다. JPEG50·블러σ2·1/2축소는 스트레스 진단이며 진행 기준에 넣지 않았다.'),'',
 f"시간: 동일96기록에서 기준 평균 {t['baseline']['mean_seconds_per_12_frame_record']*1000:.1f}ms, 후보 평균 {t['candidate']['mean_seconds_per_12_frame_record']*1000:.1f}ms. 후보/기준 합계 시간 {t['total_ratio']:.2f}배. 중앙값은 각각 {t['baseline']['median_seconds']*1000:.1f}ms / {t['candidate']['median_seconds']*1000:.1f}ms.",'',
 f"기존 기준 점수와 최대 차이 {s['max_previous_baseline_score_difference']:.3g}; 후보의 원래 화면 재추론과 기준 프레임 점수의 최대 차이 {s['max_repeated_full_frame_difference']:.3g}. 모델·실험 입력·스크립트 해시 불변 확인.",'',
 '주요3조건에서 판정이 바뀐 모든 사례:','',
 '| 원천 | 정답 | 조건 | 원래 | 반전 | 결합 | 변화 |','|---|---|---|---:|---:|---:|---|']
 for a in json.loads((P/split/'decision_changes.json').read_text()):
  if a['mode'] in ['full_frame','jpeg_75','blur_1']:
   good=(a['candidate']>=.5)==(a['label']=='recapture');lines.append(f"| {a['source']:03d} | {a['label']} | {a['mode']} | {a['baseline']:.4f} | {a['flipped']:.4f} | {a['candidate']:.4f} | {'정정' if good else '새 오류'} |")
 lines+=['']
cpu=subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True,check=True).stdout.strip()
lines+=['## 측정·해석 범위','',f'- {cpu}, CPU2스레드, 동일 모델 적재 후 워밍업. 기준과 후보를 별도로 실제 추론하고 순서를 교대했다. 시간에는 TPO 전처리·추론·평균·반전이 포함되며 공통 파일 읽기·평가 조건 변환·모델 적재·MP4 디코딩은 제외한다.',
'- 좌우 반전의 크기·dtype 보존, 첫 열/끝 열 대응, 이중 반전 픽셀 완전 복원을 self-check했다. 96개 기록 유일성, 확률 범위, 프레임 결합식, 전후 파일 해시를 확인했다.',
'- 한 재촬영 장비의 작은 거리 영상 세트이며 블랙박스 주행이 아니다. 기존 확인용은 이미 여러 실험 결과에 노출되어 새로운 독립 확인 자료가 아니다. 점수의 변화만으로 원인을 글자 방향이나 좌우 배치로 확정하지 않는다.',
'- 좌우 민감도가 관찰되더라도 평균이 분류 성능을 높인다는 결론과는 별개다. 제출 모델과 설정은 변경하지 않았다.','',
'## 재현','', '- 코드: `scripts/data/evaluate_stage1_tpo_hflip.py development`. 조정용 기준을 통과한 경우에만 `holdout` 실행이 허용된다. 기존 출력 폴더가 있으면 덮어쓰지 않는다.',
'- 결과: `artifacts/stage1_tpo_hflip_20260918/<split>/`의 protocol.json, predictions.json, summary.json, decision_changes.json.',
'- 실행 Python: `artifacts/mac_experiments/scipy_compat/.venv/bin/python`.','']
Path('docs/stage1-tpo-hflip-comparison-20260918.md').write_text('\n'.join(lines))
print('report written; development gate',passed)
