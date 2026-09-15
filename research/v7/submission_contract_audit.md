# 검증된 V6 제출본의 Stage 공통 계약 감사

2026-09-15. **이번 범위에서 새로 확인된 출력·시간축·파일 독립성 계약 오류는 없다.** 이는 정확도 보증이 아니다. 기존 Stage2 접촉 추정 및 진입 후보 문제는 새 오류로 재분류하지 않았다. TPO 전처리도 재검토하지 않았다.

대상은 workspace의 `inference.py`가 아닌 `artifacts/submissions/verify_v6/`의 실제 제출 추출본이다. 제출 manifest의 ZIP SHA는 `860f9d08d6013f090ae4bc710bdbbfeee06e01a0cd71f8a784b5689cdb4b7ee0`. 이번에 진입점·solution 전체 Python·Stage1 설정/특징 artifact·Stage3 모델/출처를 포함한 21개 파일의 SHA를 manifest와 다시 대조해 모두 일치했다. 대형 VLM/TPO 가중치와 ZIP 전체를 이번에 재해시한 것은 아니다. 코드·모델 수정, GPU 실행, 새 모델 추론은 하지 않았다.

비교 기준은 로컬 `대회_통합_정보.md:35–78,117–129,217,395`의 공식 정의 정리다. 이번 과제는 최신 웹 공지 재조사가 아니라 동결된 제출 구현과 이 정의의 대조다.

## 출력 및 번호·시간 계약

아래 코드 경로는 모두 `artifacts/submissions/verify_v6/model/stage2/code/solution/` 아래다.

| 검사 | 실제 구현과 증거 | 결론 |
|---|---|---|
| 실제 진입점 | 추출본 `inference.py:15,21,27`은 Stage1 `stage1_v4`, Stage2 `stage2_uncapped_jerk_v6c`, Stage3 `stage3_v5_compatible`를 호출 | workspace의 구형 진입점과 혼동하지 않음 |
| Stage1 범주 | `stage1_v4.py:139–145`의 고정 결합 확률 문턱 뒤 `RERECORDED`/`ORIGINAL`, ID는 영상 stem, 반환 열 `ID,answer` | 정의된 형식과 일치 |
| Stage2 프레임 번호 | `stage2.py:26`은 파일명 마지막 숫자를 원본 번호로 읽음. 정렬/내부 추론은 배열 위치를 쓰지만 `stage2_v2.py:130`에서 번호로 변환. C 최종 접촉도 `stage2_uncapped_jerk_v6c.py:83`에서 `numbers[argmax]` 반환 | 배열 인덱스 오출력 또는 일괄 ±1 보정 없음 |
| 이미 진입한 경우 | `stage2_v2.py:102–115`는 첫 이미지 후보를 제공. 첫 후보가 선택되면 실제 첫 원본 번호 반환 | 원본이 1번 시작이어도 0을 강제하지 않음. VLM이 정답 상태를 판독하는지는 별도 정확도 문제 |
| Stage2 공간/방향 | `stage2_v2.py:86–89,126–131`에서 LEFT/RIGHT 및 정수 0/1로 제한. 결측은 명시적 fallback | 출력 범주 계약 위반 없음. fallback은 관측 근거가 아니며 정확도를 보장하지 않음 |
| Stage3 클래스 매핑 | `stage3.py:9–10`: accel 0/1/2/3→ACCELERATING/DECELERATING/CONSTANT/STOPPED, steer 0/1/2→LEFT/STRAIGHT/RIGHT. 학습 코드 `research/train_stage3.py:28`도 같은 배열의 index로 라벨 생성 | 학습/추론 순서 반전 없음 |
| 실제 Stage3 저장 모델 | CPU에서 joblib 메타데이터 확인: accel Pipeline classes=[0,1,2,3], steer RandomForest classes=[0,1,2]. 키는 accel/steer/feature_version/selection | 현재 artifact에 이중 transfer regressor 경로 없음. 반환 클래스와 호환 |
| Stage3 시간 | `stage3_v5_compatible.py:110–140,150`는 source_fps=10을 지정하고 stride=1, 각 성공 디코딩마다 count 증가. 전체 count 길이로 특징 보간 | 공식 10Hz 입력과 일치. 공개 원본의 20Hz 가정을 평가 입력에 전파하지 않음 |
| Stage3 표본 번호 | `stage3_v5_compatible.py:155–159`가 두 head의 결과를 enumerate하여 0…N−1 및 두 라벨을 반환 | 시간 초/영상 PTS를 sample_index로 오출력하지 않음. 출력에 timestamp 열을 추가할 의무도 없음 |
| STOPPED 조향 | accel과 steer를 항상 독립 예측하며 STOPPED에 따른 필터/행 삭제/조향 결측 처리 없음 | 공식 GT STOPPED 마스크는 서버 채점의 역할. 제출자가 예측 STOPPED로 조향을 숨기지 않음 |

Stage3 광류 중심을 `idx−0.5`로 두고 디코딩 표본 위치에 보간하는 것은 특징 설계다. 이 내부 중심을 출력 sample_index로 사용하지 않으므로 반 프레임 출력 오프셋 오류라는 결론은 성립하지 않는다. CAN 정답과 특징의 최적 시간 정렬 여부는 별도 실증 문제다.

## 파일 간 상태와 캐시

- Stage1 `stage1_v4.py:132–145`: 고정 detector는 재사용하지만 프레임·특징·확률은 매 파일 새로 계산한다. 다른 파일의 예측이나 통계를 결합하는 코드가 없다. 고정 feature mean/scale은 학습 artifact이며 평가 파일 간 통계가 아니다.
- Stage2 `stage2_uncapped_jerk_v6c.py:32–65`: previous frame, previous shift, features, valid paths가 매 `_dual_motion_scan` 호출 안에서 초기화된다. 한 영상의 MAD/median 계산을 다른 영상에 누적하지 않는다.
- Stage2 `vlm.py:25–51`: 각 ask가 단일 user 메시지와 그 호출의 이미지만 새로 구성한다. 이전 대화·출력·past_key_values를 다음 generate에 넘기지 않는다. `use_cache=True`는 현재 생성 호출의 연산 캐시 사용이며 자체 코드에 파일 간 대화 캐시 전달은 없다. CandidateVLM은 이 ask를 그대로 상속한다. 이번 검사는 전체 라이브러리 내부 상태나 모든 입력 순서의 비트 동일성을 새로 실행해 증명한 것은 아니다.
- Stage3 `stage3_v5_compatible.py:110–116`: DIS 객체, feature computer, prev/features/centers/count를 영상마다 새로 만든다. 분위수 위치·ROI 기하 캐시는 그 영상의 feature computer 안에만 존재한다. smoothing과 보간도 그 영상 배열 안에서 수행한다.
- 공유되는 Python 모듈 캐시·고정 모델 가중치·processor는 다른 평가 파일의 정보로 답을 보정하는 코드가 아니다. Stage3 OpenCV 스레드 수는 finally에서 원복한다. 공유 설정이 있다는 이유만으로 파일 독립성 위반으로 판정하지 않았다.

## 이미 존재하는 실행 증거와 남은 한계

`artifacts/submissions/verify_v6_results/report.json`은 실제 추출본 entry point의 오프라인 실행 PASS를 기록한다. Stage1 10행, Stage2 3행, Stage3 2,998행이며 기존 비교 대상과 출력 동등성 true, 네트워크 시도 0이다. 이는 이번 새 실행 결과가 아니라 기존 검증 기록이다. `research/v5_stage3/compatible_864/summary.md`에는 파일명 변경·다른 파일 추가 시 기존 Stage3 결과 동일 및 인덱스 연속성 검사도 기록되어 있다.

확인 범위를 넘어서는 사항:

1. Stage3 디코더는 최초 read 실패에서 종료한다. 손상/일시 디코딩 실패 이후에 추가 유효 프레임이 있는 평가 파일에서의 복구는 보장되지 않는다. 현재 공식 입력에 그런 파일이 있거나 실제 행이 누락됐다는 증거는 없다. 정상 EOF와 손상을 구분하지 않는 코드만으로 이번 점수 원인을 확정하지 않는다.
2. Stage1 빠른 seek의 그럴듯한 거짓 메타데이터와 Stage2 읽기 불가 프레임 제외는 기존에 기록된 입력 한계다. 공식 평가에서 발생했는지는 확인 불가다. 명세상 정상 프레임 폴더에서의 원본 번호 보존과 구분한다.
3. V6의 최종 motion 접촉과 VLM 내부 접촉이 다르면 공간 답변은 최종 접촉이 아닌 내부 접촉 주변 문맥에서 생성된다(`stage2_v2.py:117`, `stage2_motion_collision.py:20–27`, C `:83–94`). 이것은 이미 명시된 의미상 결합 한계이며 이번 새 발견이 아니다. 어느 시점이 실제 접촉인지 모르므로 전체 공간 출력이 틀렸다고 단정하거나 이 감사만으로 수정 후보를 채택하지 않는다.
4. 실제 서버 전체 입력의 파일 수·길이·손상 여부는 비공개다. 유한 로컬 PASS는 전체 60분 통과·정확도·모든 파일 형식에 대한 보장이 아니다.

**이번 감사에 따른 즉시 production 변경 제안은 없다.** 새로 확인된 계약 오류가 없는 상황에서 클래스 순서·10Hz 기준·STOPPED 조향 처리를 바꾸는 것은 근거 없는 변경이다. 기존 정확도 문제는 관측 가능한 원인별 실험으로 별도 다룬다.
