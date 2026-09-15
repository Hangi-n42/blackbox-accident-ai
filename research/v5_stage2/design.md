# V5 Stage2 사전 구조 설계

상태: GPU 실행 전. 기존 모델·추론 소스·제출 ZIP은 보존한다. 이 설계 이후 새 `solution/stage2_v5.py`와 CPU 계약만 구현하며 GPU 실험은 root의 별도 배정/사전 동결 후 수행한다.

## V4 실패에서 확인한 것과 확인하지 못한 것

V4 공식 S2는0.217521476, V3는0.2281306982로 차이는−0.0106092222다. S1/S3는 같고 V4 전체 실행은44분33초였다. V4 첫 질문 문구가 공개 한 사례의 방향을 개선했다는 관찰은 공식 성능 개선으로 일반화되지 않았다. 공식 하위 네 점수는 없으므로 하락분을 방향·진입·공간에 분배할 수 없다.

| 근거 | 직접 확인된 사실 | 원인 해석의 한계 |
|---|---|---|
| `solution/stage2_v2.py:76`, `:90`, `:104`, `:117` | 첫 질문은 충돌/방향 동시 응답. 첫 충돌이 fine 후보를 결정하고, 다음 충돌 응답이 entry 상한과 space 이미지를 결정 | 첫 질문 문구가 방향뿐 아니라 시점/공간에 영향을 줄 구조. 실제 비공개 오답 비중은 미확인 |
| `research/v4_stage2/repro_run/report.json` S2_002 | 같은 실제 JPEG에서 V3 overview44/LEFT→fine44→entry36, D overview38/RIGHT→fine45→entry4. 최종 collision은둘다motion47 | 동일 입력에서 방향 문구 변경이 충돌 응답과 시간 탐색에 전파된 직접 사례. 어느 진입값이 공식 정답인지는 모름 |
| `research/v4_stage2/input_representation_audit.json` | 연구용 PIL JPEG와 실제 검증 OpenCV JPEG250개 모두RGB/양자화표다름. D는 S2_002에서표현별출력다름 | 작은RGB변화에 대한민감성. 같은입력재현은PASS했으므로비결정성으로단정하지않음 |
| `solution/stage2_v2.py:25` 및 `solution/vlm.py:29` | 한 번의36타일입력이아님. 실제최대10/15/12/3타일, 타일384×256중번호영역28픽셀, 전체ask예산1.2MP | 작은바퀴/차선증거를잃을위험은구조상있지만화질이공식하락의주원인이라는실측은없음 |
| `research/v4_stage2/results.md` | motion시점으로entry/space문맥을통일한A는004진입4→15로관찰과더불일치 | A를무손실최적화나V5로재포장하지않음 |
| `results.md`, `results_C.md` | B/C 추가진입질문은각5개모두BEFORE_WINDOW. 국소창과시작참조만으로이미진입한대상을복원하지못함 | 동일상대식별실패/과거진입/상태규약이해실패중원인비중은모름 |

단순 문구 재조정과 공개5개에 맞춘 frame offset은 중단한다. V5는 **상대 차량 참조를 명시하고, 진입의 전체 구간 탐색과 국소 정밀 탐색을 분리하며, 방향/공간의 언어 응답이 시간 탐색을 바꾸지 못하도록 한다.**

## 하나의 후보: 상대 참조 → 전구간 진입 경계 → 원본 ROI 정밀 선택 → 후단 분류

최종 collision_frame은 V3의 기존 motion argmax 원본 번호로 유지한다. 이는 정확한 접촉 GT가 아니라 통제할 고정 기초 정책이다. collision을 개선했다는 주장은 하지 않는다. **entry 입력 범위를 그 motion시점까지 자르지 않는다.** A와의 핵심 차이다.

| 호출 | 입력·출력 | 시간 탐색에 허용되는 영향 |
|---|---|---|
| 1 상대 식별 | 기존 motion피크2개와그직전근방의서로다른최대4원본이미지. 각이미지번호를명시. 실제충돌상대가식별되면reference_frame 및정규화bbox, 아니면UNCERTAIN | 충돌/진입/방향/공간시간값을요청하지않음. reference만고정 |
| 2 진입 coarse | 같은reference crop과전체입력구간12개균등원본프레임. 처음/끝포함. 상태BRACKET/ALREADY_AT_START/REENTRY/UNCERTAIN, before_frame/after_frame/best_frame | 특정motion상한으로입력절단하지않음. 상태·실제제시번호·시간순서검증후진입경계후보만제안 |
| 3 진입 fine | 유효경계의10개원본후보+구간밖전역anchor2개(가능하면원영상처음/끝). 잘못된coarse일때전구간12개로fallback. 각프레임의시간정합ROI와작은전체화면문맥,동일reference | OBSERVED+제시된정수번호만수용. ALREADY_AT_START는원본첫번호와해당상대확인이명시될때만수용. 구간밖/재진입/불확실/비정상은검증된coarse best_frame유지 |
| 4 방향/공간 | 고정reference,최종진입이전/진입무렵원본장면,고정motion충돌장면의전체도로시야. entry_side와evasion_space만분류 | 이미결정한reference/collision/entry/후보는수정불가. 여분시간키가반환돼도무시 |

최대4회이며 데이터셋 차원의 cache, 타 영상 응답, ID별 규칙은 없다. 파일마다reference와추적상태를초기화한다. 영상이1프레임이면유효원본을중복질문없이재사용하고필요없는fine을생략할수있다. 오류처리도추가VLM호출을만들지않는다.

### 원본 ROI의 시간 정합

reference bbox를다른시점에그대로복사하지않는다. reference이미지내특징점으로부터인접원본프레임을양방향추적한다. 기존OpenCV로작업하며320픽셀너비의회색영상,최대64개코너,Lucas–Kanade forward/backward 일치검사를사용한다. 점6개이상과과반수일치,역추적오차허용조건을만족할때만중앙변위/크기변화로bbox를갱신한다. 영상크기변화·점소실·범위오류·큰불일치이면그방향의추적을중단하고해당시점은전체원본화면으로fallback한다. 최초bbox를계속재사용하지않는다.

fine crop은시간별유효bbox에여유영역을붙여원본RGB에서추출한다. 작은전체화면thumbnail도함께보여줘차선과도로문맥을보존한다. reference선택이틀리면추적이안정적이어도다른차량을따를수있다. flow의수치검사만으로동일사고상대임을보장하지않으며별도시각검수가필요하다. bbox가없거나불확실하면ROI를강제하지않고reference전체프레임과전체화면후보를쓴다.

원본프레임번호,디코드목록위치,ROI좌표,PTS는별개필드다. normalized bbox는각원본이미지크기에매핑하고경계안으로제한한다. 검증용FPS/PTS를추론정책에주입하지않고10FPS를가정하지않는다.

### postcollision 재진입 및 잘못된 coarse guard

전체시간을보면충돌후차량의재등장을최초진입으로잘못고를위험이있다. coarse와fine은동일상대의최초진입과재진입을별도상태로구분한다. REENTRY/AFTER_WINDOW/UNCERTAIN은새시점을수용하지않는다. motion보다늦은경계나진입도유효한관측응답이면수용한다. entry>collision은진단으로만기록하며motion오류를진입의인위적상한으로전파하지않는다. 대회포맷에서entry<=collision강제를확인하지못했다.

첫설계의수치guard는독립검토후GPU전에제거했다. 후단언어응답의시간전파만차단한다. motion은여전히reference후보와space시야에영향하므로motion자체오류를해결한구조라고주장하지않는다. 유효coarse best가없으면원본첫프레임을규격유지fallback으로사용하고발생빈도를보고한다. 이는정확도보증이아니다.

fine에전역anchor를남겨coarse이웃밖의증거를완전히버리지않는다. 그래도4회·유한후보로임의길이영상의±0.3초coverage를보장할수없다. 실제oracle coverage를별도로계산해야한다.

## 렌더링·실행 예산

모델과NF4설정은동결한다. 총시각예산은ask당기존1.2MP이하이며다중이미지의균등예산분할을기록한다. 원본ROI는축소contactsheet를확대하지않고원본RGB에서만추출한다. coarse/fine은reference영역과후보영역을명시적으로배치한하나의예산내canvas를우선한다. 타일수·전체화면및ROI배율·실제bounded크기를로그에남겨'원본사용'과'원본해상도유지'를혼동하지않는다.

초기출력토큰상한은식별64/coarse96/fine48/분류48,총256으로고정한다. reference bbox응답이상한에서잘리면무한재질문하지않고fallback한다. 추적은프레임당작은회색영상2장과좌표목록만보유한다. 모델/이미지로드·추적·ROI렌더·VLM시간을따로측정한다. V4의44분33초에서남은15분27초를곧바로새연산예산으로간주하지않는다. 원영상길이분포와Stage2비중이불명확하므로추가추적비용/토큰증가를실측해야한다.

## 사전 채택 기준

1. **계약 gate:** 원본/비연속번호·offset·동영상별초기화·1프레임·bbox불량/추적소실/해상도변화·coarse순서오류·재진입·JSON불량을CPU로검증한다. 마지막분류응답에악의적시간키가있어도이미결정한time/reference/후보가같아야한다. 최대4회와시각예산을검증한다.
2. **원인 분리:** 같은입력에V3와V5를paired실행한다. reference차량일치/추적유실률/잘못된ROI비율, entry候補oracle coverage,유효후보가있는데선택실패한비율, fallback/JSON실패를나눠기록한다. 새표현이더예뻐보인다는이유만으로채택하지않는다.
3. **입력 표현 gate:** root가제공하는canonical lossless PNG,기존JPEG,mirror를사용한다. 모두동일5개원천의파생본이며독립15개가아니다. collision/entry/space의mirror불변과side반전을진단하고실패를기록한다. canonical과JPEG의차이를무시한exact gate를쓰지않는다.
4. **정답 gate:** 제공collision은그대로평가하고동일motion정책유지를확인한다. 공식entry/side/spaceGT가없으면정확도개선확정불가다. 예측미열람시각관찰/새로합법확보한검증자료와는별도라벨출처로비교한다. 공개이미진입/명확한방향사례의known regression은채택금지근거이며,관찰불명항목을정답으로강제하지않는다.
5. **채택 판단:** 위구조gate·실제동일입력패키지재현·메모리/시간gate를통과하고,현재확인가능한핵심진입/상대식별오류를고쳤다는대조근거가있어야후보로고려한다. 작은공개5개일치만으로일반화나공식향상을보장하지않는다. 근거가없으면V5 Stage2를자동채택하지않고root에한계를보고한다. 공식점수로문구·offset·bbox margin을반복선택하지않는다.

## 구현 파일 및 보존

새모듈 `solution/stage2_v5.py`, 새CPU계약 `scripts/test_stage2_v5.py`, 보고서/실험은 `research/v5_stage2/`에만추가한다. 기존stage2/entry_refine/V4모듈·가중치·ZIP·선택보고서는덮어쓰지않는다. GPU전에모델/코드/입력해시와이설계의SHA를root와동결한다.

## GPU 전 검토 반영 및 구현 상수

- Root/독립 reviewer 합의로 motion 이후 entry를 거부하는 수치 guard를 제거했다. REENTRY/NOT_FIRST_ENTRY 및 불확실 상태는 계속 거부한다. 후보가 제시됐는지와 의미상 최초진입인지는 별개다.
- 추적 최대300개 인접 step을 사전 고정한다. 전체 필요한 step이300을 넘으면 해당 영상은 reference 자체 이외의 ROI를 모두 full-frame fallback한다. 영상/시간 후보는 자르지 않는다. 시간에 따라 결과가 달라지는 wall-clock 중단은 쓰지 않는다.
- 1280×928 단일canvas=1,187,840px, reference strip160px, 후보최대12개. ROI margin35%, reference margin10%, LK 역추적오차1.5px, residual3px, 최소6코너/원래점과반수, step scale0.8..1.25, 이동48px 이하, frame경계clipping면적75% 이상이다. 이미지320px 폭의 수치이며 GPU 결과로 튜닝하지 않는다.
- observed_already_at_start / coarse_retained / invalid_fallback_first / fine_observed를 별도 기록한다. fallback 첫번호의 우연한 일치를 관측 성공으로 세지 않는다.
- 추적 안정성은 semantic identity 증거가 아니다. blind 자료의 실제 사고 상대 포함률/잘못된 ROI 비율을 별도 gate로 삼는다. motion reference 후보가 실제 충돌과 떨어져 있는 위험은 남는다.
- 같은원천표현 paired 검증에서 명확한 항목의 실제 오류 감소와 다른 명확항목 악화없음을 요구한다. 미확인 항목을 정답으로 강제하지 않는다.
- 사전 시간 gate: 공개5원천의 표현별 V5/V3 predictor 누적시간 비율이 각각1.25 이하, 최대 VRAM7.5GiB 이하, 파일별최대4call/전체120call이하를 검토 기준으로 둔다. 이 비율은60분완주증거가 아니며 실제패키지시간여유 별도검증이 필수다. gate실패시 이번후보 자동채택하지 않는다. 이는 measured 결과가 아니라 보수적인 사전 운영 기준이다.

사전시간gate 수정시각(UTC): 2026-09-13T10:24:19.937815+00:00. GPU 결과 열람 전 root 결정으로1.15→1.25를다른Stage와통일했다. V4총44분33초전체가1.25배인보수적산술은55분41.25초이지만환경/입력분포차이를보장하지않는다. 결과후문턱변경없음. 긴5000-frame CPU계약에서타임라인끝이유지되고추적decode가예산초과시발생하지않는것을검증한다.
LK 역추적은이전원본point를초기값으로준single-level LK로일치검사한다. 무작위합성translation CPU계약에서기본multi-level reverse가local역검증목적과맞지않게큰오정합을보여GPU전수정했다. semantic동일상대보증은아니다.
