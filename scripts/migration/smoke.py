"""Small CPU/import/codec checks requiring no external models or videos."""
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    import av,cv2,numpy as np,torch,torchvision,transformers
    import pandas,scipy,sklearn,joblib
    from solution.stage3_v5_compatible import OriginalFrameFeatureComputer
    cv2.setNumThreads(2);torch.set_num_threads(2)
    features=OriginalFrameFeatureComputer()(np.zeros((96,160,2),dtype=np.float32))
    assert np.isfinite(features).all() and features.size==144
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'synthetic.avi'
        with av.open(str(path),'w') as container:
            stream=container.add_stream('mpeg4',rate=10);stream.width=160;stream.height=96;stream.pix_fmt='yuv420p'
            for n in range(3):
                frame=av.VideoFrame.from_ndarray(np.full((96,160,3),n*30,dtype=np.uint8),format='rgb24')
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        with av.open(str(path)) as container:count=sum(1 for _ in container.decode(video=0))
        assert count==3
    from paths import resolve_recorded
    assert resolve_recorded('C:/Users/dsl/Desktop/Dacon/블랙박스/solution/stage3.py')==ROOT/'solution/stage3.py'
    print(json.dumps({'status':'passed','OS':platform.system(),'machine':platform.machine(),'python':platform.python_version(),
        'torch':torch.__version__,'CUDA_available':torch.cuda.is_available(),
        'MPS_available':bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()),
        'tests':['core imports','144 motion features','PyAV3frame encode/decode','legacy-path mapping'],
        'not_tested':['full Stage2 NF4 on Mac','MPS model equivalence','external asset completeness']},indent=2))

if __name__=='__main__':main()
