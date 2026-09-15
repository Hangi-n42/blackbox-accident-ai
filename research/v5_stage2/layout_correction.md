# 렌더링 수정 사전 기록

Root 승인: 최초 V5 실제 화면에서 식별4장이4열1행으로배치되어1280×928 canvas에서320×180×4=230,400px(19.4%)만사용했다. 이는세부정보를살리려던설계와불일치한다. 최초실행은51call완료+1call중단(최종clip-report48call)을보존했으며canonical5pair에서reference전부UNCERTAIN,entry전부invalid_fallback_first,시간비1.44446으로실패했다. 이결과를삭제하거나성공으로재해석하지않는다.

수정은후보3~4장의columns를2로두는배치한줄뿐이다. reference식별4장은640×360×4=921,600px(77.6%)를차지한다. 최종분류3장의도로시야도2열배치가된다. 최대12개coarse/fine은기존과같고질문/응답schema/추적조건/fallback/시간상한/픽셀/토큰예산은변경하지않는다. 모델결과를본뒤의일반배치수정이며새사전모델설계라고위장하지않는다. 부분실행후수정이므로해당관찰편향을남긴다.

CPU에서는실제source크기1280×720의4panel렌더가모두640×360인지,canvas점유율75%초과인지,3panelreference포함상황의폭600이상인지검증한다. 모델이보는PNG도실행전에직접확인한다. 원본소스/실험코드/설계/CPU결과snapshot은paired_run/source_snapshot에보존했다.

새실험은동일5source3표현V3/V5각pair,최대120call,한모델로드이다. 초기실행의소모52call은별도추가비용으로기록한다. 채택시간문턱1.25및known-regression기준변경없음. 외부DADA추론금지유지.

기록UTC: 2026-09-13T10:30:06.062634+00:00