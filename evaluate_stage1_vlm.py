"""Record fixed-prompt VLM judgments on the public Stage 1 examples."""
from pathlib import Path
import argparse
import json
import time
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from solution.stage1 import sample_video, visual_judgment, VISUAL_PROMPT
from solution.vlm import LocalVLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model/stage2/vlm')
    parser.add_argument('--data', default='Baseline/data/stage1')
    parser.add_argument('--output', default='research/stage1/vlm_judgments.json')
    args = parser.parse_args()
    data = Path(args.data)
    labels = pd.read_csv(data / 'labels.csv')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Frozen zero-shot visual evidence on public simulated examples, not actual recapture validation.',
              'prompt': VISUAL_PROMPT, 'model': args.model, 'samples': []}
    with LocalVLM(args.model, pixel_budget=1200000) as vlm:
        for row in labels.itertuples():
            start = time.monotonic()
            result = visual_judgment(sample_video(data / row.path), vlm)
            result.update({'ID': row.ID, 'label': row.label, 'seconds': time.monotonic() - start})
            report['samples'].append(result)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False), flush=True)
    truth = [r['label'] for r in report['samples']]
    pred = [r.get('answer') or 'INVALID' for r in report['samples']]
    report['macro_f1'] = f1_score(truth, pred, labels=['ORIGINAL', 'RERECORDED'], average='macro')
    report['confusion_matrix'] = confusion_matrix(truth, pred, labels=['ORIGINAL', 'RERECORDED', 'INVALID']).tolist()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
