# DA3-Small Mac 공동 궤적 특징 검사

판정: Mac에서 실행되지만 현재 21프레임/2초/504해상도 camera-decoder 기반 가감속 특징은 미채택. 분류기 학습·공식S3 평가·운영 변경 없음. 모든 DA3모델/설정의 실패로 일반화하지 않는다.

## 실행 경로와 무결성

공식 DA3-SMALL 가중치137,248,940bytes를 revision e08cab65ca0ec38e7826075418411ab90cab4da3에 고정했다. SHA256은 provenance.json에 기록·검증했다. 실제 모델은34,299,463파라미터이다. 공식코드 commit3d835ec1a5802d64a8b8b15f817a1ab54809bfe4를 수정하지 않고 공식cfg/core net/InputProcessor를 직접 사용했다. 내보내기·CUDA 전용 부가 패키지를 불필요하게 설치하지 않았다. torch2.8.0 및 기존환경은 유지했고 필요한 소형패키지만 별도deps/에 설치했다.

초기 load_state_dict(strict=True)는6개 공유 LayerNorm alias때문에 실패(exit1)했다. 공식 safetensors.load_model(strict=True)로 공유가중치를 인식하는 경로를 사용해 missing/unexpected0을 확인했다. 임의초기값을 허용한 것이 아니다. 최초오류로그·수정전스크립트를 보존했다.

MPS FP32 3프레임 smoke는3.63초, 유한한camera pose출력으로성공(exit0). 이후 whole-image21frame 입력에서 수동mask 없이 원래28창순서를 고정했다. reference first / reverse 후시간순복원 / reference middle은 한 방법의 안정성대조이며 정답에 맞는 variant를 선택하지 않았다.

## MPS 본검사: 일부 실행 후 사전 시간 상한 종료

600초 이후 새추론을 시작하지 않는 상한에 따라, 진행중추론 완료를 포함한617.56초에 종료했다. 28창 중6개는3대조를 모두 완료하고1개는2대조만 실행했다. 나머지21개는미실행. 모두기존train측 개발창이며 heldout검증에는도달하지 않았다. 프로그램 exit0과 전체28창완료를 구분한다.

완료된6창의안정성통과는0/6. 7개정순출력의q는모두음수, 역순후복원출력은모두양수였지만 이를실제분류반전으로세지않는다. 완성된6창은A2/D2/C2이다. 순서나참조점변경에따른 q차≤.02/s, 한전역Sim3정렬후RMS/span≤.1이라는사전진단기준을통과하지못했다. 이기준은대회범주임계값이아니다.

예: expanded_11 22.5초의센서중앙a/v=+.146363/s인데MPS q는정순−1.043118, 역순복원+.940704, 중간참조−1.217713이다. 등속 expanded_11 52.5초도센서−.001586에정순−1.009593/역순+1.245970/중간−1.385848이다.

camera center=-R^Tt, 시간순복원, 중앙quadratic도함수q=(v·a)/(v·v), Sim3계산과28개센서중앙a/v정의는전문가가독립감사했다. 회전/전역scale불변 합성검사도통과했다. 시각별scale오차나진짜pose정확도는이검사로확정할수없다.

## CPU 장치 대조

같은첫21프레임을CPU로추론했을때MPS와extrinsic최대차.9223, Sim3잔차비.2912가나타났다. backend일치가확인되지않았으므로MPS결과를모델자체의오류로만돌리지않는다.

이차이를확인한뒤원래목록의첫A/D/C각1창만고정해CPU에서동일3대조를시행했다(원래core/가중치/해상도/후처리유지). 첫CPU정순결과를재사용해추가8회추론했고72.91초에끝났다. CPU에서도안정성통과0/3이다.

| 범주 | 센서q | CPU 정순 | CPU 역순복원 | CPU 중간참조 |
|---|---:|---:|---:|---:|
| 가속 | +.146363 | −1.496787 | −.780269 | −2.684126 |
| 감속 | −.153632 | +1.211502 | −1.458360 | +2.827462 |
| 등속 | −.001586 | −.104479 | −.173395 | +.738872 |

CPU의대표검사도불안정하므로MPS만수정하면해결된다는근거는없다. 그렇다고CPU/MPS차이의원인연산, 입력순서효과의물리원인, 다른DA3설정의성능을확정한것도아니다. 이번범위에서backend포팅·다른해상도·큰모델·rayhead탐색을하지않는다.

신뢰할수있는새특징이확보되지않아기존DIS분류기에추가하지않았다. 새원천18창에도아직적용하지않아평가예약상태를유지한다.

주요파일: pilot_freeze.json, pilot_results.csv, pilot_summary.json, poses/, cpu_crosscheck.json, cpu_control/summary.json, expert_arithmetic_audit.json, reference_step_audit.json, pilot_expert_review.md, exit_status.json.
