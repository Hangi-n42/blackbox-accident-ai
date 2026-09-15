# V4 Stage2 최종 D 결과 및 선택 근거

root는 D를 V4 Stage2로 선택했다. A/B/C는 패키지에서 제외한다. 기존 V3의 첫 overview 질문 중 방향을 묻는 한 문장만 바꾸었으며, 기존 소스/모델/ZIP을 수정하지 않았다. D가 마지막 후보이며 결과 이후 문구 재조정이나 E 탐색을 하지 않았다.

## 사전 설계와 변경 범위

`design_D_frozen_before_results.md`와 `frozen_D.json`을 새 D 결과 전에 저장했다. 새로운 `solution/stage2_v4_d.py`는 frozen `stage2_motion_collision._predict_file`을 그대로 실행하되, 매 파일 새 wrapper로 첫 ask의 다음 문장만 치환한다.

원문: `From which side of the image did that other vehicle approach? `

치환문: `From which image side did that same vehicle originate before entering the camera car's driving corridor? Track the collision vehicle backwards. Report its image side of origin before lane entry, not its direction of travel, side of impact, or position after entry. `

첫 질문의 충돌 질문·JSON 키·64토큰 예산과 모든 입력 이미지가 같다. 기존 충돌 fine/진입/회피 질문 정책, 영상당4회, motion 식과 최종 motion 충돌 정책도 유지한다. 첫 충돌 응답이 바뀌면 뒤 후보가 달라질 수 있다는 가능성을 사전 명시했으며, 실제 공개5개에서 그 변화는 발생하지 않았다. 비공개 영상에서도 반드시 동일할 것이라고 보장하지 않는다.

C 기반 D는 구현 전에 취소했다. C가 A의 진입 변경을 그대로 보존하여 S2_004에서 이미 차로 안이라는 관찰과 더 어긋났기 때문이다. 최종 D는 V3를 직접 감싸므로 A/B/C나 추가 진입 질문을 실행하지 않는다.

## 실제 공개 5개 결과

| ID | 공통 충돌 | 공통 진입 | V3 방향 | D 방향 | 공통 공간 |
|---|---:|---:|---|---|---:|
| S2_001 | 35 | 24 | LEFT | RIGHT | 0 |
| S2_002 | 47 | 36 | LEFT | LEFT | 0 |
| S2_003 | 33 | 23 | LEFT | LEFT | 0 |
| S2_004 | 42 | 4 | RIGHT | RIGHT | 0 |
| S2_005 | 33 | 23 | RIGHT | RIGHT | 0 |

변경은 S2_001 방향 한 항목이다. root 및 별도 reviewer가 새 D 결과를 읽기 전에 한 AI 시각 관찰의 RIGHT 합의와 일치한다. 이것은 공식 방향 GT가 아니며 독립 검증셋 성능도 아니다. 이미 검토한 공개 사례를 근거로 가설을 설계했다는 선택 편향을 유지해 기록한다.

**S2_003의 LEFT는 해결되지 않았다.** S2_001/003의 기존 LEFT와 D의 LEFT/RIGHT는 모두 모델의 유효한 원응답이며 parser의 LEFT fallback에서 나온 결과가 아니다. 따라서 이번 관찰은 출력 파싱 오류를 고쳤다는 근거가 아니다. 주행 방향/충돌 위치/진입 이전 출발 방향 구분이 모든 오류의 원인이었다는 결론도 내릴 수 없다.

S2_002/004/005의 원영상 시작 방향은 시각 관찰에서 불확실하므로 정답을 강제하거나 방향 정확도를 계산하지 않았다. 진입/회피 공식 GT도 없어 하위 성능이나 최종 대회 개선폭은 확인 불가다. 기존 진입의 한계와 강한 motion이 실제 접촉과 다를 수 있다는 한계는 그대로 남는다.

## 동등성·실행 기록

- CPU mocked 계약7개 PASS: 정확한 첫 문장만 치환, 나머지 질문 보존 조건, 원본/비연속 번호·offset, 매 파일 초기화, 1프레임,4회 예산, 원문 불일치 guard. 모델 정확도 테스트가 아니다.
- 실제 GPU5개·20질문 완료. 모든 영상에서 첫 이미지 SHA/bounded크기/토큰 예산 동일, 의도한 문장만 변경됐다.
- 후속 세 질문의 prompt·입력 이미지·후보·원응답은5개 모두 baseline과 완전히 같았다. 최종 충돌·진입·공간도 모두 동일하다.
- 제공 충돌 GT를 원본 MP4 PTS에 연결한 ±0.3초 적중은4/5로 유지됐다. 제공 S2_002 라벨의 시각적 접촉 의미를 임의 수정하거나 라벨을 제외하지 않았다.
- 저장 NF4 모델의 전체 manifest 파일을 실행 전후 SHA 검증했다. 원본 source와 이전 A/B/C report도 그대로다. 네트워크 시도0, offline실행.
- RTX2080 SUPER 측정: predictor평균10.599초, 질문합평균9.538초, scan평균0.629초. 같은 대조 실행 baseline의 predictor평균10.062초였다. PNG/로그 비용이 포함된 로컬 공개5개 측정이다. 실행 순서/환경 변동이 포함되어 있으며 전체60분 통과나 추가 소요시간을 이 평균으로 보장하지 않는다.
- 최대CUDA allocated3,931,418,112bytes. 세션97143 exit0 확인 후 GPU 해제했다.

## 선택 해석

채택 근거는 대회 정의에 맞춘 명시적인 방향 질문, 시각 합의가 있었던 S2_001의 오류 수정, 공개 다른 출력·후속 질문 불변이다. 공식 GT·별도 사고 holdout·공식 채점의 개선을 입증한 것은 아니다. A/B/C는 실패 결과를 보존하고 제외한다. 최종 ZIP 무결성·오프라인 실행·실제 제출은 root가 별도로 진행한다.

## 재현 파일

- 결과 및 전체 호출 원본: `D_run/report.json`, `D_run/D/{ID}/call_*/result.json`과 PNG.
- 평가 로거는 기존 V3의 네 질문 이름을 재사용하므로 콘솔 variant가 baseline으로 표시되지만, D_run 내 모든 영상은 D 실행이다. report에 이 의미를 기록했다.
- 최종 소스: `solution/stage2_v4_d.py`, SHA256 `5dc50281d04a9310adeb5e1e42b11ff6d28ee0868828d3964ef652788e5b3e73`.
- 평가: `scripts/evaluate_stage2_v4_d.py`, SHA256 `a08d8cc1b9004f159b2406c2ff73683d3e8bfd1596326fb8f77d4ab776d772a5`.
- 테스트: `scripts/test_stage2_v4_d.py`, SHA256 `0a8ba7f8df424e9765dc0096cda74ee4f60426d023cfb5884cab0c95e18f912a`.
- 의존성: `stage2_motion_collision`, `stage2_v2`, `stage2`, `vlm_candidate`, `vlm`. A/B/C와entry_refine는 D에 필요 없다.
