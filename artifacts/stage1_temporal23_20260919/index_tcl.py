import sys,json,collections
sys.path.insert(0,'scripts/data')
import build_stage1_road_pairs as a
a.OUT=a.ROOT/'artifacts/stage1_temporal23_20260919'
u='https://www.dropbox.com/scl/fo/bxelqebf0241n1a3z13dq/AMo73x3asTnXGEqQo4D1Um0/tcl.zip?rlkey=2f1us04s3ytbf05kfflet1dqg&dl=1'
x=a.index(u,'tcl');g=collections.Counter('/'.join(k.split('/')[:4]) for k in x);print(json.dumps(g,indent=2)[:5000],flush=True)
