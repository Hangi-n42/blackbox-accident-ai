# 8B NF4 export / fresh reload 준비

`nf4_8b_io.py`는 별도 CLI다. 기존 capacity runner와 V6 자산을 수정하지 않는다. 현재 GPU 로드·실제 export·실제 reload를 실행하지 않았다. 합성 manifest·설정·수치 gate·경로·실패 시 무출력 검사의 **9개 CPU 테스트만 통과**했다(`nf4_8b_io_cpu_guard_results.json`). torch도 import하지 않은 검사다.

## 1. 개발 평가 후 별도 export

먼저 `--development-run`에 완료된9개 개발영상 report/evaluation/freeze가 있어야 한다. 첫 smoke1개와 나머지8개가 별도 실행이면 root가 출처를 보존한9개 통합 보고서를 준비해야 한다. 도구가 임의로 합치거나 검증 표본을 줄이지 않는다. 그 report의 NF4/FP16 runtime과 freeze의 원천 download_manifest SHA를 검증한다.

gate는 contact accuracy 동일, entry accuracy·known side/space Macro-F1 무하락, 그 세 변경 항목 중 하나 이상 증가를 요구한다. 분모는 양쪽 동일하고0보다 커야 한다. 실제 collision 번호도 모든영상에서 같아야 한다. 평가 report SHA가 일치해야 한다. 이 gate는 기존에 노출된 개발자료 기준이며 독립 검증이나 제출 승인이 아니다. 평가기의 계산을 재학습/변경하지 않는다. 원본 human GT를 다시 해석하지 않으며 이 도구는 평가기의 알려진 분모별 수치에 의존한다.

`--reference-run`은 같은8B 원천의 실제 완료 네 호출이 저장된 실행이다. `--reference-id`는 그중 한 영상을 명시한다. `run_capacity.py`의 `ID/call_1/record.json`, `input_0.png` 형식을 사용한다. `--development-run`과 달라도 되지만 동일 download_manifest에 바인딩돼야 한다. 호출 이미지·prompt·token상한·raw답변을 저장한다. GT 기준으로 네 호출을 새로 고르지 않는다.

실행 예시는 실제 디렉터리 값으로 대체해야 한다.

```powershell
.venv\Scripts\python.exe -I -B research/v7/nf4_8b_io.py export --source artifacts/candidates/qwen3_vl_8b_v7 --target research/v7/8b_nf4_export --development-run <완료9개개발실행> --reference-run <고정4호출참조실행> --reference-id <사전고정ID> --embedding-policy cpu_fp16 --license-text <Apache2전문파일>
```

source 검사는 고정 manifest SHA `7630c131d8759237b38c31ef7966e3d12bbdb07f8040850b6a0bb78d63d4ba17`, 공식 `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, Apache2 선언을 확인한다. 모든 원천 파일 크기/SHA, LFS digest와 작은 파일 Git blob ID를 검증한다. 실제 root 실행에서는17.5GB 파일을 읽는 비용이 있다. 이번 준비에서는 읽지 않았다.

모델은 CandidateVLM으로 한 번 로드한다. NF4/FP16/doublequant false와 full1.2MP를 유지한다. `all_gpu` 또는 `cpu_fp16` 정책을 명시하고 선택한 정책에서 reference 네 raw답변과 exact 일치를 확인한 후 같은 살아 있는 인스턴스를 저장한다. 정책 차이에 따른 불일치가 있으면 export하지 않는다. CPU 정책은 `solution/cpu_embedding.py`를 호출하고 default untied 검사를 사용한다.

`--target`은 존재하지 않는 research/v7 하위여야 한다. gate 실패·출처 실패·원문 불일치 전에는 target을 만들지 않는다. 저장 중 실패하면 `EXPORT_INCOMPLETE.json`이 남으며 재시도는 새 경로를 사용한다. 소스·기존모델을 덮어쓰지 않는다. target에는 가중치/processor 외에 원본 README, Apache2 전문, 선택적 NOTICE, 변경고지, source manifest, state schema, 고정 reference와 EXPORT_MANIFEST를 남긴다.

## 2. fresh process에서 저장본 reload

```powershell
.venv\Scripts\python.exe -I -B research/v7/nf4_8b_io.py reload --model research/v7/8b_nf4_export --output research/v7/8b_nf4_reload_check
```

저장 파일 전체의 SHA/크기와 코드·정책·출처 binding을 검증하고 저장 NF4 설정으로 한 번 로드한다. export PID와 다른 isolated 프로세스여야 한다. CPU embedding subclass는 저장되지 않으므로 manifest 정책에 따라 **로드 후 재설치**한다. state_dict의 key/dtype/shape와 embedding 전체 bytes SHA가 같아야 하며 같은 네 raw답변을 다시 비교한다. 네트워크는 socket 차단과 HF offline/local-only 설정을 사용한다. `reload_report.json`은 별도 새 디렉터리에 기록하며 export manifest를 통과 상태로 덮어쓰지 않는다.

## 한계와 남은 실제 검사

- 준비 검사는 GPU quantization serialization 지원이나8B full 입력 실행 성공을 입증하지 않는다. 실제 root 실행이 필요하다.
- raw답변 exact 비교이며 현재 LocalVLM.ask가 반환하지 않는 generated token ID 전체를 따로 수집하지 않는다. raw동등을 모든 내부logit·모든입력 동등이라고 확대하지 않는다.
- CPU policy는 초기 allGPU 로드가 성공해야 한다. 초기 OOM을 해결하는 로더가 아니다. fresh reload에도 같은 한계가 있다.
- 같은 state schema라도 전체 NF4 tensor 값을 모두 원본과 직접 대조한 것은 아니다. 파일 해시·설정·embedding 전체값·고정 raw동등을 검증한다. 실제 packed reload 동작과 다른 입력 성능은 별도다.
- 실제4호출의 모든 원문이 같아도 새 모델의 정확도, 독립 validation, ZIP 통합, 서버60분 성공을 보장하지 않는다. 이 helper는 검증을 통과한 저장본을 준비할 뿐 자동 채택/제출하지 않는다.
- 실제 export부터 reference 이미지와 기록의 원래 경로가 유지돼야 한다. 저장된 RELOAD_REFERENCE의 절대경로는 로컬 검증용이며 최종 제출 assets로 무조건 포함하지 않는다.

기반: 기존 `scripts/evaluate_stage2_4b.py`의 이미 로드한 인스턴스 저장 원칙, `research/v7/run_capacity.py`의 고정 네 호출 기록, `solution.vlm_candidate.CandidateVLM`의 NF4/FP16 로드. 새 helper의 모델 식별자는8B pinned 원천에 한정한다.
