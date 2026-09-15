# 단순화 후보 결과와 최종 Stage2 선택

Root는simple도사전known-regression gate실패로미채택확정했다. **V5 Stage2는원래V3의`solution/stage2_motion_collision.py`정책을선택한다.** ROI/simple은연구소스·원응답으로보존하고제출ZIP에서제외한다. 외부DADA추론은하지않았으며추가후보탐색을중단했다. 결정과실제V3아티팩트동일성근거는`decision.json`에기록했다.

## 적합성은 통과했으나 의미 회귀가 남았다

simple은bbox/추적/복합status를제거하고V3첫질문을그대로두었다. 전체12장entry coarse→coarse인접원본fine+시작/끝anchor→motion±2 space로4call을재배치했다. 사전동결`design_simple.md`, `gpu_freeze_simple.json` SHA `94721088d1fc93b143b37da4e411091347a20007963c032083d21f114936bc1f`.

CPU10계약PASS. GPU동일입력 V3/SIMPLE 30clip/120call한회완료,소스/모델/750이미지hash불변,offline network0,한모델로드,2 CPU thread. 첫질문의prompt/실제입력RGB/token/raw답이15pair모두같고최종side도같았다. 원본번호·독립파일·first collision응답의후속시간전파차단·validcoarse보존·192token/4call을검증했다. `paired_simple_run/report.json`, `simple_analysis.json`.

| 표현 | V3 predictor초 | SIMPLE predictor초 | 비율 | 1.25 기준 |
|---|---:|---:|---:|---|
| canonical PNG |44.364|43.580|0.9823|PASS|
| 기존 JPEG |43.270|42.874|0.9908|PASS|
| mirror PNG |47.856|47.152|0.9853|PASS|

최대reserved VRAM5.623GiB로7.5GiB기준PASS. 이결과는실행적합성통과이며공식점수개선증거가아니다. 같은5원천의3표현을독립15표본으로계산하지않는다.

진입coarse/fine는15입력모두제시된유효정수답이었다. `fine_valid_selected`15/15, invalidcoarse/fine0, `invalid_fallback_first`0이었다.004의각표현0은실제모델의유효선택이며ROI후보의기계적fallback0과다르다. 유효정수라는사실만으로의미상정답은아니다.

| 원천 | PNG V3→simple | JPEG V3→simple | mirror V3→simple |
|---|---:|---:|---:|
|001|24→16|24→16|18→18|
|002|36→4|36→31|31→34|
|003|23→12|23→12|25→25|
|004|11→0|4→0|4→0|
|005|23→16|23→16|23→16|

공식entry GT는없다. 사전에저장한AI관찰에서001/003의진입advisory구간은20..30/20..31이며정확한GT또는±0.3초정답으로취급하지않는다. simple은PNG/JPEG001·003에서그관찰보다더이른시점을선택했다. 반면004는원래첫프레임에같은차로앞차가있다는관찰과맞는0을선택했다. 일부개선과다른사례회귀가함께있어Root가사전회귀억제기준에따라채택을중단했다.

003에서전체coarse후보에는22/27/31이있었지만모델은13을골랐다. `solution/stage2_v5_simple.py:24`가다음후보를이전/다음균등index사이로만좁혀fine은0,9..18일부,49가되며advisory20..31을전부잃었다. 그러고fine은12를선택했다. **전체구간을한번보였다는것만으로정밀탐색의coverage를보장하지못하며,시작/끝anchor도중간의틀린coarse창을항상복원하지못한다.** 이경로는실제후보로그로확인했다. 차선/상대식별이왜13을유발했는지의기여도는분리실험없어확인불가다.

002는PNG4/JPEG31로표현에따라크게달랐고003은원본12/mirror25였다. 입력정책의점단위독립성과의미판단의표현강건성은다르다. 제공collision은모든표현에서V3/simple4/5로같다(최종motion정책유지). side도기존첫질문보존으로같았다. space입력은15pair중14개가달랐으나최종space는전부0으로같았다. 공간문맥변경confound가사라진것이아니며비공개자료에서도같다고일반화하지않는다.

## V3 복원 경로와 근거

선택파일:`solution/stage2_motion_collision.py`

SHA256:`091d1b4b53353cf08bc5431fb658d1c7fcc45604a4288546f3aac0fd6a207e2e`

이파일과`stage2.py`, `stage2_v2.py`, `vlm.py`, `vlm_candidate.py`5개는실제V3압축해제본`artifacts/submissions/verify_v3_motion_fast/model/stage2/code/solution`의파일과각각SHA가같다. NF4모델19파일도같은실제V3아티팩트와모두SHA가같다. 따라서원래V3첫질문·4call·기존내부VLM충돌문맥의다른3필드·최종motion collision정책을복원하는선택이다. 이선택은V3가최적이라는증명이나미래공식성능보장이아니다.

새ZIP작성/제출은root담당이며이문서는이미새ZIP이완성됐다고주장하지않는다. 현재두신규구조의실패를외부자료추가실험으로살리거나공개답에맞춰수정하지않는다. 모델/소스/실패실험전체는후속연구의원인감사근거로남긴다.
