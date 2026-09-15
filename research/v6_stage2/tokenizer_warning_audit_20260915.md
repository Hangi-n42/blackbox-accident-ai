# V5 로딩 경고의 범위 확인

전체 프레임 진단 실행 중 나온 `incorrect regex pattern` / `fix_mistral_regex=True` 경고를 로컬 설치 소스와 실제 패키지 설정으로 확인했다. 이 경고만으로 Qwen 토크나이저 손상이나 낮은 정확도의 원인을 확정할 수 없다. 이번 진단에서는 설정을 변경하지 않았다.

패키지 config.json의 model_type은 qwen3_vl, transformers_version은 4.57.6이고 tokenizer_class는 Qwen2Tokenizer다. tokenizer.json의 Split 정규식은 설치된 transformers/models/qwen2/tokenization_qwen2.py의 PRETOKENIZE_REGEX와 일치한다.

로컬 transformers/tokenization_utils_base.py의 Mistral 보정 분기는 config의 transformers_version이 4.57.2 이하일 때에만 non-Mistral 모델을 제외하고, 5.0.0 이상일 때에도 반환한다. 그 사이 버전인 이 패키지는 non-Mistral 모델인데도 경고 분기에 들어간다. 경고 분기 자체는 fix_mistral_regex를 False로 두며, True를 명시한 다른 분기에서만 정규식을 바꾼다. 현재 V5는 이 플래그를 전달하지 않는다.

따라서 경고 문구를 따라 Mistral용 정규식으로 바꾸는 수정은 이번 Qwen 진단에서 근거가 없다. 이 확인은 토크나이저 전체의 정확성 인증이 아니라 해당 경고가 발생한 코드 경로에 대한 감사다. 가중치·토크나이저·추론 코드는 그대로 유지했다.
