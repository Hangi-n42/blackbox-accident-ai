# Round2 다운로드 준비 및 접촉 전용 검토

`download_sources.py` 작성 후 문법 검사만 수행했다. 다운로드·디코드·프레임 열람·모델·라벨 파일 접근은 실행하지 않았다. root가 검사 후 실행한다.

```powershell
.venv/Scripts/python.exe -I -B research/v6/nexar_review_round2/download_sources.py
```

고정 manifest SHA는 `26f9316348a1f24f93714856afca5f1205dd90528e669f1024170ad67ac91aad`다. 고정 revision의 00008/00010/00013만 받으며 합계 42,322,975바이트다. 기존 파일·partial·보고서가 있으면 시작 전에 중단하고 덮어쓰지 않는다. 각 `.part`에 선언된 크기 이내로 스트리밍하고 정확한 크기/SHA 확인 후 최종 MP4로 이름을 바꾼다. 실패 시 이미 완료한 소스별 기록과 부분 파일을 보존한다. 자동 재시도 또는 다른 영상 대체는 없다.

기존 동일 revision의 LICENSE/README를 바이트 그대로 복사하고 ATTRIBUTION.json에 출처·SHA를 남긴다. 영상에서는 픽셀을 내보내거나 열람하지 않고 PyAV 디코드 순번과 native PTS를 기록한다. PTS가 없거나 유한하지 않거나 단조 증가하지 않으면 실패한다. 각 `.mapping.json`과 소스별 acquisition 기록, 최종 `acquisition.json`을 생성한다. 원래 manifest의 선정 상태는 수정하지 않는다.

## 기존 검토 도구 CLI

`research/v6_review_tool/prepare_case.py`의 필수 인자는 `--video`, `--id`, `--source-group`, `--source-uri`, `--exposure {seen,unseen}`다. `--output` 기본값은 `research/v6_review_tool/dist`이며 `--resume`은 완료되지 않은 case만 허용한다. 완료된 `case.json`은 덮어쓰지 않는다.

00008 예시(root가 그룹 식별자를 확정한 후 실행):

```powershell
.venv/Scripts/python.exe -I -B research/v6_review_tool/prepare_case.py --video research/v6/nexar_review_round2/00008.mp4 --output research/v6_review_tool/dist --id NEXAR_REVIEW_00008 --source-group nexar_round2_00008_unverified --source-uri https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/aa97deda5a59f00bb7187739053b7c72e14374df/train/positive/00008.mp4 --exposure unseen
```

`--exposure unseen`은 case의 split을 `unassigned`로 기록한다. 이름이나 SHA가 다르다는 사실만으로 다른 사고가 인증되는 것은 아니므로 source-group 예시는 미검증임을 명시했다. 00010/00013도 영상·ID·그룹·URI를 각각 해당 고정 source로 바꿔 사용한다. 이 검토 도구는 전체 PNG를 내보내며 cases.js를 갱신한다. 이는 다운로드 코드와 별도 단계다.

## 접촉만 기록하고 다른 항목을 unknown으로 둘 수 있는가

**가능하다.** `dist/app.js:17–19`의 저장 조건은 검수자와 접촉 상대 설명이 비어 있지 않은지뿐이다. 접촉·진입·방향·공간의 완전 기입을 요구하지 않는다.

- 접촉을 관측하면 현재 프레임 기록으로 원본 순번과 native PTS를 저장한다(`app.js:12`). 실제 접촉을 판정할 수 없으면 접촉도 판정 불가로 남긴다.
- 차선 진입은 미기록으로 두거나 판정 불가를 눌러 `status: uncertain, frame: null, pts_seconds: null`로 남길 수 있다.
- 방향과 회피 공간은 기본 선택인 미기록/판정 불가를 유지하면 빈 문자열로 저장된다(`index.html:15,20`). 이는 확정 범주가 아니며 후속 평가에서 unknown으로 처리해야 한다.
- 저장물은 항상 `record_type: human_review_draft`, `evaluation_eligible: false`다. 접촉 전용 기록을 전체 Stage2 GT나 독립 다수 검토의 정답으로 승격하지 않는다.

검토자가 보지 않은 모델 예측이나 다른 사람의 답을 대신 입력하지 않는다. 기존 evaluator는 접촉 observed만 평가하며 누락된 다른 세 항목을 성능 점수로 사용하지 않는 구조다. Round2 평가에서는 새 source/주석 SHA와 원본 PTS를 별도로 연결해야 한다.
