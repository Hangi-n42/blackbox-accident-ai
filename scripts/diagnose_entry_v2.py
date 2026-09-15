"""Public entry-only diagnostic using prior model collision predictions, no new labels."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solution.stage2_v2 import _sheet, _uniform_indices
from solution.stage2 import _frame_number
from solution.vlm import LocalVLM

PROMPT = (
    "These frames are chronological, left to right then top to bottom. "
    "Track the vehicle that contacts the camera car in the FINAL image. "
    "Which numbered frame first shows that vehicle's wheel reaching the camera car's lane? "
    "If it was already inside this lane at the first image, choose the first image. "
    "Which image side did it approach from? "
    "Return JSON with entry_frame and entry_side (LEFT or RIGHT)."
)

if __name__ == "__main__":
    source = ROOT/"artifacts/eval_stage2/v2_neutral_entry_diagnostic/report.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    with LocalVLM(ROOT/"artifacts/model/stage2/vlm") as vlm:
        for video in report["videos"]:
            paths = sorted((ROOT/"research/stage2_public/images"/video["ID"]).glob("*.jpg"), key=_frame_number)
            collision = next(index for index,path in enumerate(paths) if _frame_number(path)==video["prediction"]["collision_frame"])
            candidates = _uniform_indices(0, collision, 12)
            image = _sheet(paths, candidates, columns=4)
            started = time.perf_counter()
            answer = vlm.ask([image], PROMPT, max_new_tokens=64)
            row = dict(ID=video["ID"], collision_prediction=video["prediction"]["collision_frame"],
                       prompt=PROMPT, answer=answer, seconds=time.perf_counter()-started,
                       candidates=[_frame_number(paths[index]) for index in candidates])
            rows.append(row)
            print(json.dumps(row), flush=True)
    (ROOT/"artifacts/eval_stage2/v2_entry_independent_diagnostic.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
