"""Controlled decoder fault injection: frame identity, count and explicit failure."""
from pathlib import Path
import sys,json,argparse
from unittest.mock import patch
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from solution import stage1

CASES=['normal','short','negative_count','nan_count','underreported_count','overreported_count',
       'failed_seek','seek_returns_true_but_wrong_position','seek_lies_about_position','one_seek_read_failure','persistent_selected_retrieve_failure','empty']


def factory(case):
    n=5 if case=='short' else 0 if case=='empty' else 50
    class Capture:
        def __init__(self,path):self.pos=0;self.last=-1;self.seeked=False;self.released=False;self.claimed=0
        def isOpened(self):return True
        def get(self,key):
            if key==cv2.CAP_PROP_FRAME_COUNT:
                return {'negative_count':-1,'nan_count':float('nan'),'underreported_count':20,'overreported_count':90}.get(case,n)
            if key==cv2.CAP_PROP_POS_FRAMES:return float(self.claimed+1 if case=='seek_lies_about_position' else self.pos)
            return 0.
        def set(self,key,value):
            self.seeked=True
            self.claimed=int(value)
            if case=='failed_seek':return False
            if case in ['seek_returns_true_but_wrong_position','seek_lies_about_position']:return True
            self.pos=int(value);return True
        def grab(self):
            if self.pos>=n:return False
            self.last=self.pos;self.pos+=1;return True
        def retrieve(self):
            if self.last<0:return False,None
            if case=='persistent_selected_retrieve_failure' and self.last==22:return False,None
            return True,np.full((8,12,3),self.last,dtype=np.uint8)
        def read(self):
            if not self.grab():return False,None
            if case=='one_seek_read_failure' and self.seeked and self.last==22:return False,None
            return self.retrieve()
        def release(self):self.released=True
    return Capture,n


def evaluate(module,case):
    capture,n=factory(case)
    expected=np.unique(np.linspace(0,n-1,min(12,n)).round().astype(int)).tolist() if n else []
    try:
        with patch.object(module.cv2,'VideoCapture',capture):frames=module.sample_video('mock.mp4')
        actual=[int(f[0,0,0]) for f in frames]
        return {'expected':expected,'actual':actual,'exact_uniform_selection':actual==expected,'error':None}
    except Exception as e:
        return {'expected':expected,'actual':None,'exact_uniform_selection':False,'error':type(e).__name__+': '+str(e)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',action='store_true');args=p.parse_args()
    candidate=None
    if args.candidate:from solution import stage1_v4 as candidate
    rows={case:{'old':evaluate(stage1,case)} for case in CASES}
    if candidate:
        for case in CASES:
            rows[case]['candidate']=evaluate(candidate,case)
            result=rows[case]['candidate']
            if case=='empty':
                assert result['error'] and result['error'].startswith('RuntimeError:'), (case,result)
            elif case in ['underreported_count','seek_lies_about_position']:
                assert not result['exact_uniform_selection'] and result['error'] is None,(case,result)
                rows[case]['known_limit']='Undetectable using plausible metadata/position reports without a full decode'
            elif case=='persistent_selected_retrieve_failure':
                assert result['error'] is None and len(result['actual'])==11,(case,result)
                rows[case]['known_limit']='Persistent selected-frame damage remains partial; candidate does not introduce whole-stage failure'
            else:assert result['exact_uniform_selection'],(case,result)
    name='candidate_contract.json' if candidate else 'original_contract.json'
    Path(__file__).with_name(name).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps({k:{m:{'n':len(v['actual']) if v['actual'] is not None else None,'exact':v['exact_uniform_selection'],'error':v['error']} for m,v in row.items() if isinstance(v,dict)} for k,row in rows.items()},indent=2))


if __name__=='__main__':main()
