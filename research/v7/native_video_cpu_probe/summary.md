# Native video CPU processor 계약

실제 저장4B NF4 package의 AutoProcessor만 사용했고, 모델·weight tensor·GPU는 로드하지 않았다. 합성384×256 labeled tile K3/15/18로 모두 통과했다. 기존9개 영상이나 GT는 입력하지 않았다.

| 원본 K | encoded K | grid T,H,W | video placeholder 토큰 | 총 input 토큰 | 패딩 포함 pixels |
|---:|---:|---|---:|---:|---:|
| 3 | 4 | 2,16,24 | 192 | 266 | 393,216 |
| 15 | 16 | 8,12,20 | 480 | 667 | 983,040 |
| 18 | 18 | 9,12,18 | 486 | 699 | 995,328 |

총 input 토큰은 이 검사에서 사용한 합성 질문·불규칙 번호에 한정한다. 실제 GPU 호출의 질문과 번호에 따라 달라지며 helper가 다시 기록한다.

홀수K는 정규화된 마지막 temporal patch의 두 슬롯이 비트 단위로 같았고 짝수K는 달랐다. 모든 원본 후보가 tensor에 남았으며 processor sampling/resize는 껐다. metadata는 합성 fps2, index0..K−1, total_num_framesK다. 원본 번호는 타일 픽셀에만 남겨 그 번호를 합성시간으로 변환하지 않았다. 입력 metadata의 별도 사본을 보존해 processor의 홀수 index list 확장에 영향을 받지 않는다.

예산 분모는 encoded_K=2×ceil(K/2)이며 총1.2MP에 패딩을 포함한다. 기존 LocalVLM의 BICUBIC/32배수 축소 계산식을 사용하되 분모가 다르므로 기존 모자이크·multi-image와 동일 pixels/tokens의 순수 포장 비교라고 주장하지 않는다. 빈입력·다른타일크기·18개초과를 거부하는 검사도 통과했다.

API: `ask_native_video(vlm, labeled_tiles, prompt, max_new_tokens=48) -> (raw, diagnostics)`.

root runner가 원질문의 모자이크 공간순서 첫 문장을 정확히 제거하고 나머지를 전달한다. helper는 다음 고정 문자열만 앞에 추가한다.

> Frames are ordered chronologically. The playback timestamps are synthetic and do not represent physical capture time; answer using the original frame numbers printed on the images.

diagnostics에는 actual video_grid_thw, encoded/padded frame수, tile/pixel budgets, input/video/generated token수와 생성token IDs, 합성timestamp 문자열, 타일SHA, processor/generation 시간, CUDA 실행 시 call별 peak allocated/reserved가 들어간다. helper는 GPU call 직전에 peak counter를 초기화하므로 전체 run peak는 각call의최댓값으로 집계해야 한다. CPU 검사에서는 이 GPU 분기를 실행하지 않았다.

기존 tokenizer regex 경고는 발생했으나 설정을 변경하지 않았다. 이 검사는 영상의 시각 의미 이해, 출력정확도,실제GPU메모리·시간을 입증하지 않는다. GPU 결과 이후 FPS·prefix·tile 크기를 탐색하지 않는다.

`report.json`은 runner 호환 요약이며 helper SHA와 원본CPU보고서SHA를 포함한다. 자세한 입력·token·timestamp·패딩 검사는 `../native_video_processor_probe/report.json`과 해당 `frozen_plan.json`에 남아 있다. 원래 실행 파일/동결 기록은 변경하지 않았다.
