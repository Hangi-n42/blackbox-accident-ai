from prepare import *
import shutil
rows=read(O/'results.json'); fixed=read(O/'fixed_inlier_results.json')
notes={
 'expanded_11_225':'나무·건물 외에 하늘·구름과 오른쪽 인접 차량에도 선택점이 있다. 시점별 선택점 집합이 달라진다.',
 'expanded_14_215':'배경점이 적고 도로·연석, 차체/유리 하단 무늬, 주변 차량에 점이 섞인다.',
 'expanded_19_275':'초반 검은 인접 차량과 차체/유리 하단에 점이 몰리고, 중후반 나무·건물·일부 하늘로 선택점 집합이 크게 바뀐다. 내부 기하 검사 통과가 자차 이동량 정확성을 보장하지 않았다.',
 'expanded_14_555':'차체/유리 하단 경계의 인라이어가 많고 정지 배경점은 드물다.',
 'expanded_11_525':'주변 차량, 도로 표시, 차체 부위에 인라이어가 섞인다. 등속 센서 구간에서 양의 q가 나온다.',
 'expanded_09_335':'도로 표시와 차체/유리 하단, 오른쪽 가까운 차량에 선택점이 집중된다.',
 'zod_000002_140':'초반 오른쪽 가까운 트럭, 이후 하늘·구름·나무에 점이 섞인다. 선택점 집합이 크게 바뀐다. Kannala 렌즈 모델과 무왜곡 핀홀 가정이 다르다.',
 'zod_000026_161':'도로 하단·차체 경계와 먼 가드레일에 점이 분포한다. 영상상 도로는 대체로 직선이다. Kannala 렌즈 모델과 무왜곡 핀홀 가정이 다르다.'}
write(O/'visual_review.json',{'reviewer':'AI visual review; not human or official labels','reviewed_windows':8,'reviewed_frames_per_window':[0,10,19],'unique_source_frames_reviewed':24,'all_168_frames_visually_reviewed':False,'meaning':'Green inliers denote PnP consistency, not certified static scene points. No pixel-level semantic ground truth created.','rows':[{'key':k,'evidence':f'evidence/{k}.jpg','observation':v} for k,v in notes.items()]})
summary={'windows':8,'comma_windows':6,'zod_windows':2,'unique_input_frames':168,'depth_calls':24,'inference_seconds':sum(x['seconds'] for x in read(O/'inference_log.json')),'depth_intrinsics_stable':sum(r['depth_stability']['pass'] for r in rows),'q_stable':sum(r['q_stable'] for r in rows),'all_gates_valid':sum(r['valid'] for r in rows),'fixed_inlier_q_span_within_0_02':sum(r['q_fixed_span']<=.02 for r in fixed),'fixed_inlier_span_reduced':sum(r['q_fixed_span']<r['original_q_span'] for r in fixed),'classifier_fits':0,'S3_measured':False,'independent_evaluation':False,'protected_hashes_unchanged':all(sha(R/p)==h for p,h in read(O/'freeze.json')['protected'].items())}
assert summary['protected_hashes_unchanged'];write(O/'summary.json',summary)
write(O/'exit_status.json',{'prepare.py':0,'geometry.synthetic before inference':0,'infer.py':0,'geometry.py':0,'evidence.py':0,'audit.py':0,'fixed_inliers.py':0,'meaning':'Commands completed successfully; feature quality gate failed. No classifier or official score evaluation was run.'})
license_text=(P/'SOURCE_LICENSE').read_text()+'\n\n'+(Z/'SOURCE_LICENSE.txt').read_text()+'''\n\nCurrent derived artifacts: eight selected windows, resized frames, depth/confidence maps, DIS correspondences, PnP diagnostics, AI visual overlays, and sensor comparisons. No source labels were modified. Existing dataset-specific licenses remain separate; this notice does not relicense the combined directory. ZOD-derived material retains CC BY-SA 4.0 attribution/share-alike requirements.\nDA3 source: https://github.com/ByteDance-Seed/Depth-Anything-3\nExisting DA3-SMALL weights: model revision e08cab65ca0ec38e7826075418411ab90cab4da3; code revision 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4. Apache-2.0 code license reproduced in DA3_LICENSE. No new model download.\nOriginal ZOD full license page: ../stage3_zod_20260920/vendor/ZOD_LICENSE_PAGE.html\n'''
(O/'SOURCE_LICENSE.txt').write_text(license_text);shutil.copyfile(D/'vendor/Depth-Anything-3/LICENSE',O/'DA3_LICENSE')
labels={0:'가속',1:'감속',2:'등속'}
table='\n'.join(f"| {r['key']} | {labels[r['label']]} | {r['sensor_q']:+.5f} | {r['features']['first']['q']:+.5f} | {r['features']['reverse']['q']:+.5f} | {r['features']['middle']['q']:+.5f} | {'통과' if r['depth_stability']['pass'] else '실패'} |" for r in rows)
ftable='\n'.join(f"| {r['key']} | {r['original_q_span']:.5f} | {r['q_fixed_span']:.5f} |" for r in fixed)
report=f'''# Stage3 추가 영상 단서: DA3 depth + DIS + PnP 검증

결론: **8구간 중 사용 가능한 특징 0구간. 학습 확장 중단, 채택하지 않음.** 실행 실패가 아니라 특징의 물리적 타당성·안정성 실패다. 기존 모델, 특징, 정답, 분할, 운영 파일은 보존했다. 분류기 학습 0회, S3 측정 없음, 다운로드 없음. 따라서 점수 상승도 하락도 측정한 실험이 아니다.

## 왜 이 방법을 검증했는가

화면 이동량은 실제 속도뿐 아니라 배경까지의 거리·회전·렌즈·주변 물체 이동에 좌우된다. 기존 DIS와 분류기만 반복 조정하지 않고, 깊이를 이용해 2D 대응점을 3D로 복원하고 자차 상대 이동량을 구하는 후보를 시험했다. 이전에 실패한 DA3 직접 카메라 궤적 출력은 사용하지 않았다.

공식 [DA3 구현](https://github.com/ByteDance-Seed/Depth-Anything-3)과 로컬 고정 소스에서 입력 전처리, depth의 z축 정의, 참조 프레임 재정렬 후 원래 순서 복귀를 확인했다. [OpenCV PnP 문서](https://docs.opencv.org/4.10.0/d5/d1f/calib3d_solvePnP.html)에 맞춰 3D–2D 대응으로 회전·이동을 계산했다. 이 자료들은 방법의 구성 근거이며, 가감속 점수 개선의 증거가 아니다. DA3-Small 출력을 실제 미터 단위 깊이라고 취급하지 않는다.

## 고정 범위와 실행

- 기존 comma 6구간(가속 2·감속 2·등속 2), ZOD 기존 오류 등속 2구간. 각 21프레임, 총 168 입력 프레임. 모두 개발 노출 자료이며 독립 평가가 아니다.
- 센서로 선택한 기존 구간과 센서 시각을 유지했다. 센서로 추론 결과를 보정하지 않았다. 원본 경로·시각·센서 배열은 `manifest.json`에 있다.
- 기존 DA3-Small CPU FP32, 공식 504 resize. 정순/역순 후 복원/중간 참조의 3조건 × 8구간 = 24회. 추론 합계 {summary['inference_seconds']:.1f}초. 새 가중치·패키지 설치 없음.
- 기존 DIS_FAST, 양방향 대응검사, 고정 K, EPNP RANSAC와 LM. 대응점은 깊이 조건 간 동일하다. K는 정순 예측의 구간 중앙값으로 모든 조건에 고정했다. 알려진 센서 pose나 ZOD calibration을 입력에 주지 않았다.
- q = 시간에 대한 log(상대 이동속도)의 기울기, 단위 1/s. 센서도 동일 시간창 log(속도)의 기울기와 비교했다. 양수는 속도 증가, 음수는 감소 방향이다. 대회 범주 정답이나 m/s² 자체가 아니다.
- 구간 전체 깊이 배율은 q에서 상쇄되지만 프레임별 배율 변화는 상쇄되지 않는다. 프레임별 정규화로 실패를 숨기지 않았다.
- 임계값은 평가 전 `freeze.json`에 고정했다. 실험용 품질 기준이지 공식 대회 기준이 아니다. 실제 영상 정답에 맞춘 임계값 탐색은 하지 않았다.

## 결과

깊이·내부파라미터 순서 안정성 4/8 통과. 이동 기반 q의 3조건 차이 ≤0.02/s는 0/8 통과. 종합 사용 가능 0/8이므로 유효 표본 MAE는 계산 불가(null)다. 아래 수치는 **품질 미통과 진단값**이며 분류 성능이 아니다.

| 구간 | 센서 범주 | 센서 q | 정순 q | 역순 q | 중간 참조 q | 깊이/K 안정성 |
|---|---|---:|---:|---:|---:|---|
{table}

정순의 가속 2·감속 2구간에서 센서와 q 부호가 모두 반대였다. 분류기를 실행하지 않았으므로 이것을 새 분류 반전 건수로 세지 않는다. 등속에서도 큰 양·음의 q가 발생했다.

## 원인 분리 대조

같은 DIS 대응점·K를 사용하는 최초 실험에도 PnP의 인라이어와 유효 프레임 쌍은 조건에 따라 달라질 수 있다. 따라서 사후 원인 진단으로 정순의 인라이어·유효 쌍·초기 자세까지 고정하고, 깊이값만 바꿔 같은 반복해법으로 재추정했다. 새 깊이 추론이나 임계값 탐색은 없었다. 이 대조의 다른 깊이 조건이 품질 검사를 통과했다고 주장하지 않는다.

| 구간 | 기존 q 범위 | 점/유효쌍/초기값 고정 후 q 범위 |
|---|---:|---:|
{ftable}

8/8에서 변동 범위가 줄었다. 그러나 고정 후에도 7/8은 0.02/s를 넘었다. 안정성 기준을 만족한 `expanded_11_525`도 센서 q=-0.00015에 추정 q 약 +0.225~+0.235였다. **점 선택 변화는 흔들림의 일부를 설명하지만, 이를 고정해도 참값 오류가 해결되지 않는다.** 깊이 변화에 대한 민감성과 기하·선택점의 편향이 남는다. 이 대조는 시간별로 하나의 동일한 3D 점 집합을 추적한 실험이 아니며, 원인별 기여율을 산출하지 않았다.

### 직접 확인한 근거와 확인하지 못한 부분

1. 8구간 × 3시각의 원영상·깊이·인라이어 오버레이를 AI가 검수했다. 24개 원본 프레임 표본 검수이며 전체 168프레임 수동 검수가 아니다. 주변 차량·차체·하늘에도 인라이어가 존재했다. PnP 인라이어는 기하식 적합을 뜻할 뿐 정지 배경 정답이 아니다. 픽셀별 정지/동적 정답이나 오염률은 만들지 않았다.
2. `expanded_19_275`에서 선택점은 초반 인접 차량/차체 중심에서 나무·건물 등으로 바뀌었다. 센서는 감속인데 추정 상대 이동량은 중간부터 크게 증가했다. 상관된 관찰이며, 해당 객체만 제거하면 해결된다는 인과 결론은 아직 없다.
3. 깊이/K는 4/8에서 입력 순서·참조 변경 검사를 실패했다. 통과한 구간도 q가 틀렸다. 깊이 안정성 하나만으로 자차 가감속 정확성을 인증할 수 없다.
4. ZOD 원본 calibration에는 두 사례 모두 Kannala 카메라 모델과 비영(非零) 왜곡이 명시되어 있다. 이 시험은 추정 K와 무왜곡 핀홀 근사이므로 ZOD 오류를 DA3만의 실패라고 결론낼 수 없다. comma calibration도 검증하지 않았다. 세부 값은 `calibration_review.json`에 별도 보존했다. ZOD의 정답 calibration만 제공해 얻는 성능은 대회 영상 적용 가능성과 별개다.
5. 센서 비교치는 기존 정렬·평활화 자료의 대체 기준이다. 8개의 반복 노출 개발 구간으로 전체 자료나 다른 depth/odometry 방법의 실패를 일반화하지 않는다.

## 구현과 보존 검증

정확한 합성 3D–2D 대응에서 회전·이동·가감속 방향 복원, 전체 깊이 3배 배율의 q 불변을 확인했다. 복원된 단계별 속도에 시간 가변 배율을 주면 q가 그 배율의 log 기울기만큼 변하는 것도 확인했다. 이는 수식/솔버 검사이며 실제 영상 정확성 증명이 아니다.

`audit.py`는 24개 q·깊이 변화 통계를 재계산하고 24개의 대표 프레임 쌍 PnP를 재실행했다. 보호 대상 원본·정답·운영 모델·고정 스크립트 해시는 일치했다. 모든 실행 명령 exit 0 (`exit_status.json`). 특징 채택 판정은 실패 (`decision.json`).

## 다음 진행 판단

**현재 DA3-Small + DIS + 무왜곡 PnP 조합의 학습 확대·제출은 하지 않는다.** 같은 공개 정답에 맞춘 깊이 배율 보정이나 q 임계값 탐색도 하지 않는다.

다음 한 번의 저비용 원인 대조는 기존 프레임에서 차량·차체·하늘을 제외한 **수동 검수 정지 배경 대응점**으로 원래 점 선택만 교체하는 것이다. 깊이·K·해법은 고정하고 센서 q의 부호/오차와 안정성을 비교한다. 이는 실제 제출 특징이 아니라, 정지 배경 자동 선별에 투자할 가치가 있는지 확인하는 상한 진단이다. 배경점이 부족한 구간은 성공으로 세지 않고 결측으로 둔다. 배경만 써도 실패하면 segmentation 모델 도입이나 분류기 학습으로 확대하지 않는다.

렌즈 왜곡은 그 다음에 따로 다뤄야 한다. ZOD 제공 calibration을 사용하는 진단은 가능하지만, 대회 영상에 같은 정보가 없으므로 그대로 실험 투입 가능한 특징이라고 주장할 수 없다. 정지점 선별과 렌즈 보정을 동시에 바꾸지 않는다. 현재 결과만으로 특정 새 모델이 점수를 개선할 것이라는 근거는 없다.

## 산출물

`manifest.json`, `freeze.json`: 입력·설정·보호 해시. `depth/`, `flow/`: 재사용 가능한 깊이·대응점. `results.json`, `decision.json`, `summary.json`: 지표와 판정. `fixed_inlier_results.json`: 원인 대조. `evidence/`, `visual_review.json`: 검수 근거. `SOURCE_LICENSE.txt`, `DA3_LICENSE`: 출처·이용 조건 보존. `audit_checks.json`, `synthetic_checks.json`, `exit_status.json`: 실행 및 보존 검사.
'''
(O/'REPORT.md').write_text(report)
print(json.dumps(summary,ensure_ascii=False,indent=2))
