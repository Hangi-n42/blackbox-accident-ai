# Stage2 공개 PNG/JPEG 움직임 선택 대조

동일한 V7 배포 소스의 V6C CPU 함수에 canonical PNG와 현재 JPEG를 각각 입력했다. VLM/GPU/MLX 호출·학습·다운로드·운영 파일 수정은 없다.

| 입력 | 최종 프레임 | 접촉 Acc@0.3초 | MAE |
|---|---|---:|---:|
| canonical | [35, 43, 33, 43, 33] | 4/5 | 0.460000초 |
| current_jpeg | [35, 47, 33, 43, 33] | 4/5 | 0.540000초 |

선택이 달라진 영상: ['S2_002']. 기존 캐시 승자 재현 여부는 results.json의 항목별 기록에 보존했다. 점수 byte 동일성과 승자 동일성을 구분한다.

## 해석과 한계

- 입력 표현만 달리한 동일 CPU 실행에서 선택 차이가 재현되는지 확인하는 감사다. JPEG 압축만의 효과와 디코더·생성 경로 차이는 분리하지 않았다.
- canonical PNG 250장은 과거 동결 해시, 현재 JPEG 250장은 9/17 입력 manifest 해시에 모두 일치했다. 현재 JPEG와 과거 v5 manifest의 existing_jpeg는 같은 경로여도 별도 입력이다.
- 현재 JPEG가 과거 existing_jpeg 해시와 같은 수: 0/250.
- prepare_public_eval.py는 OpenCV 순차 디코딩과 기본 JPEG 인코딩 경로를 제공한다. 현재 JPEG를 생성한 실제 명령·시각·코덱 버전은 이번 감사로 확정하지 않았다.
- PNG의 원본 RGB/native PTS 대응은 기존 motion_public_audit 기록을 근거로 한다. 이번에는 원본 영상 SHA를 확인했으며 원본 영상을 다시 디코딩하지 않았다.
- 공개5개는 이미 노출된 개발자료이고 제공 접촉 부분 라벨만 평가했다. 다른 세 필드·공식 S2·비공개 일반화 개선을 주장하지 않는다.
- 과거 캐시와 현재 연산 사이의 작은 수치 차이는 기록했다. 연산 라이브러리/플랫폼 원인을 추가로 분리하지 않았다.

## 재현

프로젝트 루트에서 다음 명령을 사용한다. 기존 결과는 덮어쓰지 않으며 새 하위 폴더를 지정한다.

```sh
PYTHONDONTWRITEBYTECODE=1 artifacts/mac_experiments/scipy_compat/.venv/bin/python artifacts/stage2_goal_20260919/representation_audit/run.py --output-dir artifacts/stage2_goal_20260919/representation_audit/reproduction_01
```

manifest.json: 실행 전 입력·코드·환경 해시. results.json: 모든 점수·특징·픽셀 차이·캐시 대조. manifest 바인딩은 계산 완료 후 재확인했다.
