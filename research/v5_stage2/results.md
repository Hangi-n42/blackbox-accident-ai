# V5 상대 참조·ROI 구조: 채택 불가

새 구조는 CPU 형식 계약을 통과했지만 의미 판단과 사전 시간 기준을 통과하지 못했다. 공개 원천5개를 PNG/JPEG/좌우반전으로 만든15입력에서 진입이 전부 `invalid_fallback_first` 경로의0으로 귀결됐다. 이것은15개 독립 검증이 아니며 이미차로안 관찰과0이우연히일치하는경우를개선으로계산하지않는다.

## 실행과 보존

- 최초실행은식별4장의전체canvas점유가19.4%인배치문제를실제PNG에서발견하여중단했다. 51call완료+1call중단, clip단위report는48call까지저장됐다. `paired_run`, `gpu_freeze.json`, `interrupted_run_audit.json`, `paired_run/source_snapshot` 보존.
- Root 승인으로3~4장일때2열로배치하는소스한줄만수정했다.4장의실제패널은640×360,canvas점유77.6%다. 질문/응답schema/추적조건/픽셀/토큰/시간gate는바꾸지않았다. 부분결과후배치수정이라는절차를숨기지않는다. `layout_correction.md`.
- 새freeze SHA `cd9f9e2bcdc8887cfc151d133d6a40432619eddcdb1487860e67fc4ac209ff4c`; 최종후보 SHA `620911e7ff91b1518938dace444a3517a5c666bc8e8461d7b78cbe7d36f83af1`.
- 새실행은정확히동일입력 V3→V5 pair,3표현×5원천×2모델경로=30clip/120call. 한NF4모델로드,2 CPU thread,HF offline/socket차단. source/model/750입력hash전후동일,network0. GPU프로세스정상종료·해제.
- 실제제시PNG, call별raw JSON/prompt/tokens/시각예산/시간/VRAM,원본번호/bbox/추적중단사유: `paired_layout_run/report.json` 및하위폴더. 후처리집계 `results_analysis.json`.

## 사전 gate 결과

| 표현 | V3 predictor초 | V5 predictor초 | V5/V3 | 1.25 시간gate | IDENTIFIED 형식응답 |
|---|---:|---:|---:|---|---:|
| canonical PNG |47.253|84.276|1.7835|실패|3/5|
| 기존 JPEG |45.653|70.999|1.5552|실패|1/5|
| mirror PNG |46.081|66.912|1.4521|실패|0/5|

predictor시간은렌더링/추적/VLM을포함하고저장PNG·hash·로그의측정오버헤드는제외했다. 모델로드와공통motion scan은별도다. 최대reserved VRAM5.260GiB로7.5GiB기준은통과했다. 4call이라는수만으로기존비용유지를보장하지않았다. bbox/상태출력으로첫질문생성시간이늘었고마지막분류canvas도기존3타일보다크다. 이측정은서버60분완주보증이아니다.

CPU16계약PASS: 마지막side/space응답이time/reference를바꾸지못함, motion보다뒤인유효entry수용,원본·비연속번호/offset,독립파일,1프레임,JSON/상태불량,추적소실/크기변화,원본ROI이동,렌더배치,5000frame전체타임라인보존과trackingbudget검증. 별도실파일CPUstress에서는합성동일texture301 PNG의300step을1.507초에추적했고302 PNG에서예산초과로추적decode0회/fullframe fallback했다. 이는실제영상대표runtime나semantic정확도가아니다. `cpu_contract.json`, `tracking_cpu_stress.json`.

## 의미 판단이 실패한 경로

`solution/stage2_v5.py:192`의coarse수용은상태/원본번호/순서가유효해야한다. 실제coarse15/15는REENTRY였다. fine15/15도수용하지못했다. 결과적으로`solution/stage2_v5.py:285`의분기에서**invalid_fallback_first15/15, observed_already_at_start0/15, fine_observed0/15, coarse_retained0/15**다. 정상적인JSON파일과허용정수출력은나왔지만모델의의미판단성공은아니다.

| 원천 | V3 PNG entry | V3 JPEG entry | V3 mirror entry | V5 모든표현 entry |
|---|---:|---:|---:|---:|
| S2_001 |24|24|18|0 (무효fallback)|
| S2_002 |36|36|31|0 (무효fallback)|
| S2_003 |23|23|25|0 (무효fallback)|
| S2_004 |11|4|4|0 (무효fallback)|
| S2_005 |23|23|23|0 (무효fallback)|

진입GT는공식제공되지않았다. 예측전에저장된root의AI관찰에서001/003은첫프레임부터들어와있는상대가아니며,진입advisory구간은각20..30/20..31이었다. 두원천모두coarse/fine에구간내후보가존재했다. 따라서현재실패를단순히'12개격자에후보가없었다'고설명할수없다. **후보가있어도상대/최초진입상태를판별하지못하고불확실성을첫프레임으로바꾼다.** 해당AI구간은정확한공식GT나±0.3초정답이아니다. `results_analysis.json`의AI_advisory_only필드에coverage를분리했다.

V5는entry_side LEFT, evasion_space1을15입력모두반환했다. mirror의time/space동일5/5는상수fallback에서도생길수있으므로강건성증거가아니다. side반전은0/5였다. 방향이명확하다는기존AI합의001/003 RIGHT를수정하지못했다. 공간공식GT는없으므로space1이얼마나오답인지계산하지않는다.

## 잘못된 reference의 강한 고정

IDENTIFIED4응답을곧바로성공4건으로세지않았다. 원본/참조crop을시각검토한결과는별도`roi_visual_audit.json`에기록한다(AI의사후관찰,공식bbox GT아님).

- canonical001 frame42 bbox는도로·노란ego후드를주로포함하고오른쪽사고SUV본체를보존하지못했다. 이영역의기하추적은19프레임에서수용됐다.
- canonical002 frame43 bbox는하늘/전봇대/수평선영역으로주황트럭을포함하지못했다.기하추적은9프레임에서수용됐다.
- canonical/JPEG004 frame42 bbox는앞차의하단일부/번호판과ego후드위주다. 상대의일부는보이지만차체·바퀴·차선판단참조로충분한지는확인불가다. 명확한완전식별성공으로계산하지않았다.

`solution/stage2_v5.py:263`은형식상유효bbox를reference로확정하고`:257`이후질문은이를동일사고상대라고명시한다. LK검사는인접영상의국소정합만확인한다(`:56`,`:111`). **잘못된region도안정적으로추적될수있으며,그것을뒤의질문이신뢰할참조로강제한다는취약점이직접관찰됐다.** 실제정확한상대를찾아전달하는단계가성립하지않으면객체참조추가는오류고정을강화한다. bbox좌표계오해/차량식별실패/복합출력과부하각각의기여는분리실험없어확인불가다.

제공collision은각표현에서V3/V5모두4/5(원본PTS±0.3초)로동일하다. 최종기존motion argmax를의도적으로유지했으므로V5시각판단의향상근거가아니다. 또한002의motion선택은PNG43과JPEG47로달라졌다. representation민감성은VLM에만한정되지않는다.

## 판단

현재V5 ROI후보는채택하지않는다. 형식/원본번호/격자탐색/예산계약PASS와의미성능PASS를구분해야한다. 이번실험은후단언어응답의시점전파를차단한구조계약은증명했지만,상대참조가틀렸을때의오류전파및불확실성→첫프레임편향을해결하지못했다. 사전시간gate도모두실패했다. 기존제출·모델은보존했고새학습/제출/외부DADA추론을하지않았다.

다음단일단순화의검토는`simple_followup_review.md`에만기록하며추가구현·실행은별도승인전진행하지않는다. 작은공개5원천을맞추는threshold/문구사후탐색은하지않았다.
