# V5 단순화 후보 사전 동결 설계

ROI후보의실패(잘못된참조고정/복합상태15회REENTRY→0fallback/시간gate모두실패)에대응하는Root승인단일단순화다. 공개답을맞추는frame별규칙이나threshold탐색을하지않는다. 이안이실패하면추가후보탐색을중단한다. 기존소스/ROI실험/가중치/ZIP은보존한다.

1. V3첫질문·원본renderer·10장·64tokens를정확히유지한다. 방향응답과fallback도같다. 이call의collision값은모든후속후보와최종time에사용하지않는다.
2. 원영상전체의균등12장,원본첫/끝포함,진입단일entry_frame JSON. 40tokens. 복합status/bbox없음. motion이entry범위상한이되지않는다.
3. coarse의이전/다음균등후보사이9개+coarse자체+원본첫/끝의최대12장으로정밀선택. 동일단일entry_frame JSON,48tokens. 원영상시작anchor와국소창시작의차이를명시한다. coarse가비정상이면전구간12개fallback. 실제제시정수만수용하고비정상fine은검증된coarse보존.
4. evasion_space질문은V3원문과40tokens를유지하되입력문맥을최종motion argmax의전후2개원본목록index로고정한다. 마지막응답의time/side키는무시한다.

최대4call,합192tokens,LocalVLM 기존1.2MP budget/NF4/모델/원본번호정책유지. 1프레임도4call이고추가재질문없음. 좌표ROI/추적/상태schema없음. 원본FPS가정없고비연속번호·offset·독립파일입력계약유지.

이안은원래최종collision에는쓰이지않던VLM collision질문을진입정밀탐색에재배분한다. V3의내부collision이space문맥에쓰였으므로space이미지동등성은성립하지않는다. motion±2문맥고정이라는confound를명시하고space변화는독립진단한다. 실패한V4A와달리entry입력을motion에서자르지않지만정확한동일상대식별이나거짓coarse창문제는남는다.

fallback경로는fine_valid_selected / coarse_valid_retained / invalid_fallback_first로분리하고valid_model_selected_first를별도기록한다. 유효0응답도영상관측이진실이라는보증이아니며실제진입AI관찰/공식GT는분리한다. 모든이미지/원응답/후보/시간/메모리를기록한다.

사전gate: CPU계약PASS 후소스/테스트/runner/설계/19model파일/750inputhash동결. V3/SIMPLE의동일입력pair를canonical→JPEG→mirror순서,각5원천에서총30clip/120call한회실행한다. 한모델로드,2 CPU thread,offline/socket차단. predictor시간비각표현≤1.25,VRAM≤7.5GiB. 동일5원천파생15개를독립표본으로간주하지않는다.

정확한collision은기존motion정책동일을확인하고제공PTS±0.3초로만평가한다. entry/side/space공식GT없음을유지한다. 미리저장된명확한AI관찰의known regression이없고진입핵심오류를고쳤다는근거가있어야채택후보로삼는다. 잘모르는대상의side/space를정답으로강제하지않는다. 공간변경confound와first-frame fallback우연일치는향상근거가아니다. 결과후시간문턱/문구/후보규칙을수정하지않는다.

현재외부DADA12개는예측하지않는다. 전체공개조건을통과한뒤Root의blindlabel/기준합의가있을때만별도실험한다. 공개5개좋은결과만으로일반화/공식점수향상을확정하지않는다.
