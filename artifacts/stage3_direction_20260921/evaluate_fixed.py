"""Serialize NumPy scalar gates; frozen formulas, predictions and settings unchanged."""
import json
import numpy as np
import run

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))

if __name__ == '__main__':
    run.check_time()
    run.roi.write = write
    run.evaluate()
