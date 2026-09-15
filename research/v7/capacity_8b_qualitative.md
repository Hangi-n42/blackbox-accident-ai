# 원래 V6 정책에서 4B·8B 원응답 대조

2026-09-15. 노출된 개발 9개에 대한 읽기 전용 독립 검토다. 모델·GPU 실행, 코드·라벨 변경, 신규 시각 주석은 하지 않았다. 이 문서는 root의 정식 집계·자원·채택 판정을 대신하지 않는다.

**결론: 8B의 유효한 JSON 생성은 확인되지만, 이 V6 정책에서 의미 판단이 개선되었다는 근거는 없다.** 방향은 9/9 RIGHT로 단일 클래스에 모였고, 회피 공간은 4B와 동일하게 9/9 0이다. 내부 충돌은 양쪽 모두 사용자 접촉 ±0.3초에 0/9, 진입은 양쪽 모두 1/8이다. 최종 충돌 불변은 VLM 성능 결과가 아니라 동결된 motion 후처리의 결과다.

## 원응답과 fallback

| 점검 | 4B V6 | 8B V6 |
|---|---|---|
| 방향 원응답 | RIGHT 5, LEFT 4 | RIGHT 9, LEFT 0 |
| 회피 공간 원응답 | integer 0: 9 | integer 0: 9 |
| 알려진 방향 7개와 일치 | 3/7: 00004·00007·00010 | 3/7: 00004·00008·00013 |
| 알려진 공간 9개와 일치 | 2/9: 00000·00013 | 동일 |
| 진입 ±0.3초 일치 | 1/8: 00010 | 동일 |
| 내부 VLM 접촉 ±0.3초 일치 | 0/9 | 0/9 |
| collision fine 후보가 접촉 ±0.3초 포함 | 5/9 | 동일한 5/9 |

각 모델의 36개 응답은 모두 JSON으로 직접 해석된다. 방향과 공간은 원응답의 유효 클래스가 그대로 최종값이고, collision fine·entry의 정수도 실제 제시 후보에 속한다. 따라서 이번 단일 클래스 출력은 잘못된 JSON을 LEFT/0으로 바꾼 fallback 현상이 아니다. 특히 8B의 RIGHT는 기본 fallback인 LEFT와 다르다. 00003·00006의 사용자 방향 공란과 00004 진입 불확실은 정답으로 채우지 않았다.

8B는 00008·00013 방향 일치를 얻는 대신 00007·00010 일치를 잃었다. 모든 답을 RIGHT로 바꾼 결과이므로 이 두 개선 사례만으로 상대 차량의 진입 방향을 더 잘 구분한다고 주장할 수 없다. 00003은 시작부터 같은 차로에 있었다는 사용자 진입 0에 대해 4B 492, 8B 583으로 둘 다 실패했다. 00004의 진입 971→109는 변화가 크지만 사용자 진입이 불확실하므로 개선·악화로 판정하지 않는다.

## 내부 시점과 최종 충돌은 서로 다르다

`artifacts/submissions/verify_v6/model/stage2/code/solution/stage2_v2.py:104`는 내부 VLM collision까지 12개 진입 후보를 만든다. 같은 파일 117행은 그 내부 collision ±2프레임으로 공간 질문을 구성한다. 이후 `stage2_uncapped_jerk_v6c.py:83`은 최종 collision만 uncapped motion argmax로 바꾼다. 그러므로 최종 collision이 맞아도 entry·space 질문이 실제 접촉 주변을 보았다는 뜻은 아니다.

- **00000:** 4B·8B 모두 coarse 1028, fine 1093, entry 596, RIGHT, space 0으로 원응답 네 개까지 같다. 사용자 contact 584에 대해 내부 collision은 약 +17.612초다. 최종 motion 584가 맞는 것을 8B의 접촉 인식 성공으로 계산하면 안 된다.
- **00003:** coarse 668→1202, fine 601→1068로 바뀌었다. 사용자 contact 582 대비 내부 오차는 +0.633→+16.2초다. 8B fine 후보에는 실제 접촉 허용 구간이 없고, 최종 motion은 두 모델 모두 583이다. 내부 시점 오류와 최종 출력 성공이 동시에 존재한다.
- **00004:** fine 1068→1201, 내부 오차 +16.3→+20.733초다. 정답 허용 구간의 후보가 실제 fine 입력에 있어도 늦은 프레임을 골랐다. 원응답만으로 그 이유가 대상 오인인지 사건 의미 오인인지 확정할 수 없다.
- **00010:** 내부 fine은 620→621로 여전히 접촉 637의 ±0.3초 밖이다. entry 564→565는 양쪽 모두 사용자 entry 572의 허용 구간 안이다. 최종 collision은 두 모델 모두 초기화 아티팩트가 의심된 index 1을 유지한다. 모델 용량만 바꾼 이번 조건은 최종 motion 선택을 고치지 않는다.

fine coverage가 있는 5개는 00000·00004·00005·00007·00013이며 양쪽 모두 그 안에서 접촉 시점 선택에 실패했다. 나머지 4개는 이번 fine 입력 자체에 접촉 허용 구간이 없다. 모델 교체만으로 후보 누락과 후보 내 선택 실패가 모두 남았다.

## 비교 범위와 근거

V6 정책 자체는 `research/v7/solution/stage2_capacity_v7.py:5`에서 frozen V6 `_predict_file`을 그대로 호출한다. 첫 질문 문구는 9개 모두 4B 기록과 동일하다. 그러나 모델의 앞선 응답이 바뀌면서 두 번째 질문의 실제 후보 목록 문구는 5개, 세 번째 질문은 7개에서 달라졌다. 네 번째 문구가 같아도 내부 collision이 바뀌면 제시 영상은 달라진다. 따라서 동일 정책 비교이며, 네 질문의 실제 픽셀을 전부 고정한 비교가 아니다. 이번 검토에서 렌더 이미지를 새로 판독하지 않았으므로 원응답에서 대상 차량 정체의 의미적 정확성을 별도로 인증하지 않는다.

원응답 자료:

- 8B: `research/v7/capacity_8b_smoke/00000/result.json`, `research/v7/capacity_8b_remaining/{00003,00004,00005,00006,00007,00008,00010,00013}/result.json`의 `calls`·`diagnostics`·`baseline`·`candidate`·`frame_pts`.
- 4B: 기존 6개는 `research/v6_stage2/v5_fullframe_diagnostic/traces/{ID}/trace.json`, 추가 3개는 `research/v6_stage2/round2_validation/run/report.json`. 최초 6개의 최종 V6 collision은 8B 결과에 바인딩된 기존 V6 `baseline`으로 확인했다. 옛 V5 capped 최종 collision과 혼동하지 않았다.
- 사용자 검수: `research/v7/stage2_dev_run1/evaluation.json`이 지정하는 기존 9개 `review_path`. 이번 읽기에서 파일 SHA가 해당 `review_sha256`과 모두 일치함을 확인하고, bound native PTS에서 허용 오차를 대조했다. 검수 파일과 정답은 변경하지 않았다.
- 8B smoke report SHA: `eba2bd18c77802a40eee279f1af81d44a8aa392ac3e2078d34ce27055d1cfeea`.
- 8B remaining report SHA: `43f03ba51a84455106e1d246dfc26cd034e72ae76085dba3a013184a5ee4bf8d`.

읽은 두 보고서는 `status=complete`, network attempts 0이다. 이는 새로 전체 모델 자산을 재해시하거나 서버 실행을 검증했다는 뜻은 아니다. 정확도·시간·VRAM에 대한 공식 채택은 별도의 사전 gate로 결정해야 한다. 8B 크기나 precision만으로 품질 향상을 주장하지 않으며, 이 결과를 아직 실행하지 않은 8B×unified 조건의 결과로 대신 해석하지 않는다.
