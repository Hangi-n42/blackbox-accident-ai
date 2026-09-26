# 기존 검수 화면 재사용 경로

기존 도구 `/Users/hyeongi/projects/blackbox-accident-ai/research/v6_review_tool/prepare_case.py`를 그대로 재사용할 수 있다. 신규6개를 기존 dist에 합치지 말고 이 acquisition 하위 별도 viewer 디렉터리를 목적지로 쓰면 기존 사례·초안과 분리된다. 이번에는 전체PNG 생성·viewer 등록·브라우저 실행은 하지 않았다.

`prepare_case.py`가 생성하는 `case.json` 필수필드: ID, source_group_id, video_path, video_sha256, source_uri, split, exposure, frame_number_rule, time_origin, mapping_method, frames, labels=null. 각 frames 항목은 frame, pts_seconds, image, sha256다. `cases.js`는 `window.REVIEW_CASES = [...]` 형식이다. 새 디렉터리에 기존 dist의 index.html/app.js/style.css/NEXAR_LICENSE.txt도 함께 복사해야 한다.

예시 명령(프로젝트 루트에서; 아직 미실행):

```sh
PYTHONDONTWRITEBYTECODE=1 artifacts/mac_experiments/scipy_compat/.venv/bin/python research/v6_review_tool/prepare_case.py \
  --video artifacts/stage2_goal_20260919/acquisition/00024.mp4 \
  --output artifacts/stage2_goal_20260919/acquisition/viewer \
  --id NEXAR_GOAL_20260919_00024 \
  --source-group UNVERIFIED_NEXAR_00024 \
  --source-uri https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/aa97deda5a59f00bb7187739053b7c72e14374df/train/positive/00024.mp4 \
  --exposure seen
```

이6건은 모델예측 미노출이나 AI선별에는 노출됐으므로 기존도구의 보수적 seen/development로 등록하고 추가 provenance를 별도 보존한다. `unseen` 플래그만으로 독립평가를 인증하면 안 된다. PNG전체덤프는 원본99.4MB보다 커지는 별도 로컬 저장비용이므로 필요한 사례부터 등록한다. 기존도구는 매프레임 원본PTS와 PNG SHA를 생성하며 기존case를 덮어쓰지 않는다.

`human_review_templates/*.json`은 아직 빈 양식이므로 기존 검수완료 importer에 넣지 않는다. 실제 사람이 viewer로 내보낸 파일은 human_review_draft/evaluation_eligible=false이며 후속 독립검수·조정 전까지 확정GT가 아니다.
