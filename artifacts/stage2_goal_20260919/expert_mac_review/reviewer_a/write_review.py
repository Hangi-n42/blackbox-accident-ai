import json
from pathlib import Path
O=Path('/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_a')
j=json.loads((O/'records.json').read_text())
lines=['# Stage2 원본 독립 AI 검수 A','',
'이 문서는 **AI의 독립 시각 검수**이다. 사람 전문가가 작성·확정한 정답이 아니다. `human_review=false`, `model_predictions_seen=false`, `other_reviews_seen=false`, `model_inference_run=false`로 기록했다. 00024–00029 원본 영상, 각 PTS 매핑, 이 원본에서 생성한 blind 시트·native PNG만 사용했다. 기존 정답·초안·모델 후보·점수·다른 검수는 읽지 않았다.','',
'**결과: 여섯 영상 모두 최초 실제 접촉을 확정할 시각 증거를 확보하지 못했다.** 충돌이 없었다는 결론은 아니다. 근접·제동·자세 변화 장면은 확인되지만, 실제 접촉면이 가려지거나 화면 밖에 있다. 그러므로 이 검수만으로 `collision/entry/direction/space`의 강제 정답을 생성하지 않았다.','',
'관측한 차량 추적과 실제 사고 상대의 확정은 별도이다. 특히 00028의 검은 SUV가 첫 원본 프레임부터 같은 차선 안에 있다는 사실은 직접 보인다. 그 SUV가 실제 충돌 상대라고 별도 확정되는 경우에만 대회 규칙상 `competition_entry=f0, PTS=0, 0초`를 적용할 수 있다. 물리적 최초 진입은 시작 이전 시점 미상이다.','',
'## 확인 범위','',
'전 영상의 시작부터 끝까지 0.5초 이하 간격으로 관찰했다. 00028의 원본 속도는 30fps와 달라 공유 명목 2Hz 시트 일부 간격이 0.522876초였다. 해당 간격에 24개 중간 프레임을 추가로 추출·관찰하여 최대 간격을 0.490196초로 낮췄다. 이외 5개 영상의 최대 간격은 0.5초다. 사건 주변은 대략 10Hz 시트, 선택 구간은 모든 native 프레임, 주요 프레임은 native PNG로 직접 판독했다.','',
'원본 전체를 디코딩해 세부 추출 시 각 프레임의 실제 native PTS가 제공 매핑과 일치하는지 확인했다. 여섯 영상의 현재 SHA-256도 PTS 파일의 원본 SHA-256과 대조해 모두 일치했다. 음성은 검수하지 않았다. 전체 원본 모든 프레임을 연속 재생한 검수로 표현하지 않는다.','',
'| ID | 직접 본 고유 프레임 | 전체 시간 범위 | 근접/시야 변화 재검토 구간 | 실제 최초 접촉 |','|---|---:|---|---|---|']
for r in j['records']:
 c=r['review_coverage']; i=r['visual_incident_candidate']['interval']; a,b=i['start'],i['end']
 lines.append(f"| {r['id']} | {c['unique_directly_viewed_frame_count']} | 0–{r['source']['last']['time_s']:.6f}초 | f{a['frame_id']}–f{b['frame_id']}, {a['time_s']:.6f}–{b['time_s']:.6f}초 | unknown |")
lines += ['', '위 구간은 근접·화면 자세 변화를 보고 독립적으로 선택한 **추정 재검토 구간**이다. 충돌 발생을 확정한 구간도, 최초 접촉의 상·하한도 아니다. 충돌 존재 자체가 미확정인 상태에서 임의의 넓은 범위를 `collision=inferred_interval`로 만들어 기록하지 않았다.','', '총 직접 관찰 고유 프레임은 772개다. 구체적인 프레임 목록, native PTS, time base, 근거 시트 경로와 개별 PNG 경로는 [records.json](records.json) 및 ID별 JSON에 전부 보존했다.','']
for r in j['records']:
 v=r['id'];lines += [f'## {v}', '',r['observed_vehicle_track']['description']['value'],'']
 fields=r['competition_targets']
 for title,k in [('접촉','collision'),('진입','entry'),('방향','direction'),('공간','space')]:
  lines.append(f"- **{title}: unknown.** {fields[k]['uncertainty_reason']}")
 i=r['visual_incident_candidate']['interval'];a,b=i['start'],i['end']
 lines += ['',f"근접/시야 변화 재검토 범위: f{a['frame_id']} / PTS {a['native_pts']} → f{b['frame_id']} / PTS {b['native_pts']}, time_base={a['time_base']}. {r['visual_incident_candidate']['value']}",'','주요 직접 시각 근거:','']
 for e in r['physical_contact_surface_directly_visible']['evidence']:
  p=Path(e['image_path']);f=e['frame']; rel=p.relative_to(O) if p.is_relative_to(O) else p
  lines.append(f"- [f{f['frame_id']} · PTS {f['native_pts']} · {f['time_s']:.6f}초]({rel}) ({e['view_method']})")
 lines += ['','직접 본 세부 범위:','']
 for x in r['review_coverage']['dense_inspections']:
  fs=x['frames']; ids=[z['frame_id'] for z in fs]
  lines.append(f"- `{x['tag']}`: {len(ids)}개 프레임. f{ids[0]}–f{ids[-1]}; 정확한 선택 목록은 [{v}.json]({v}.json)의 `review_coverage.dense_inspections`에 기록.")
 lines += ['']
lines += ['## 사용 조건','',
'- `observed`는 직접 보이는 차량·위치·상태에만 사용했다. 관측 차량의 LEFT/RIGHT를 실제 충돌 상대의 대회 direction으로 자동 승격하지 않는다.',
'- `inferred_interval`은 본 기록에서 근접·시야 변화의 검토 구간에만 사용했다. `is_collision_time_interval=false`를 명시했다.',
'- `unknown`은 음성 정답, 사고 미발생, 또는 검수 실패가 아니다. 원본의 관측 한계를 설명하는 유효한 결과다.',
'- 실제 사고 상대와 접촉 존재·시점을 뒷받침하는 추가 증거 없이 이 결과를 사람 정답 또는 학습·평가용 강제 정답으로 사용하지 않는다.','',
'전체 기록: [records.json](records.json). ID별 구조화 기록: '+', '.join(f'[{r["id"]}.json]({r["id"]}.json)' for r in j['records'])+'.','']
(O/'review.md').write_text('\n'.join(lines))
print(O/'review.md')
