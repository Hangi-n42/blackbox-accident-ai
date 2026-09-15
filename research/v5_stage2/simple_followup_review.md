# 후속 단순화 검토 — 아직 구현·실행하지 않음

Root요청에따른설계검토다. 현재ROI구조의관찰실패는유효bbox를의미상정확한상대로오인한것,복합entry상태15/15 REENTRY→첫프레임fallback,생성·시각비용증가다. 공개답을보고특정frame·offset·threshold를고치는대신,입출력요구를V3의짧은단일정수선택수준으로되돌리는가설은이증거와직접관련된다.

4call내한후보는다음과같다.

1. V3첫overview질문과이미지/토큰을그대로유지하여side를얻는다. 그collision응답은entry후보선택에전달하지않는다.
2. 원영상전체12개균등프레임에서동일사고상대의entry_frame하나만고르게한다. 첫/끝포함,원본번호유지. motion으로entry범위를자르지않는다.
3. coarse의인접원본후보구간을정밀하게제시하되coarse자체와원본첫프레임/끝anchor를유지한다. 같은단일entry_frame JSON으로선택하며관측된원본정수만수용한다. 비정상답은검증된coarse를유지한다. local창첫이미지가fullclip시작이아님을명시한다.
4. space는사전에명시한고정문맥정책으로판단한다. side/space가entry시점을다시바꾸지못한다.

이안은bbox/추적/여러상태필드가없어참조오식별을강제로고정하지않고추가decode/긴JSON비용을줄인다. 실패한V4A는entry입력을motion까지잘랐지만이안은전구간→국소탐색이라는차이가있다. 다만복잡상태를없애면불확실/재진입을명시적으로표현할능력도줄며,잘못된차량을고르거나coarse창밖실제진입을놓칠위험은남는다. 단순정수답이항상나온다는것을정확도개선으로보면안된다.

**V3와space문맥동등성은그대로유지할수없다.** 기존call2의VLM collision은최종collision출력에서는버리지만space입력선택에는사용한다. 그call을entry로바꾸면원래내부collision을얻을수없다. motion±2혹은call1coarsecollision기준등새정책을사전에하나고정해야하며,space변화는독립confound로기록해야한다. 이전개별영상의V3답을cache해일반추론에섞으면파일독립규칙과정책동등성해석을훼손한다. 4call예산에서원래space의완전보존을주장하지않는다.

가장정보량있는검증은새공개원천의blind counterpart/진입관찰에서(1)전체coarse에정답이보이는지,(2)fine에남았는지,(3)실제로맞는후보를골랐는지의분해다. 같은5원천파생은회귀·표현진단만가능하다. 사전known-regression/timegate를다시명시하고,단일정수fallback도관측선택/invalid fallback경로를구분해야한다. 현문서는후보채택이나효과를확정하지않는다.

후속 상태: 이 검토 기록 이후 root가단일simple구현/paired실험을승인했다. 사전동결설계는design_simple.md, 결과는paired_simple_run에별도로보존한다. 이문서의미실행표기는검토작성당시상태다.
