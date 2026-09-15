"""Synthetic source fixtures only. Does not open any real review or model artifact."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
from PIL import Image

spec=importlib.util.spec_from_file_location('round2_evaluator',Path(__file__).with_name('evaluate_validation.py'))
evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)


class SyntheticNativeContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='round2_native_fixture_')
        self.folder=Path(self.temp.name);source=self.folder/'synthetic.mkv'
        with av.open(str(source),'w') as output:
            stream=output.add_stream('ffv1',rate=10);stream.width=32;stream.height=24;stream.pix_fmt='bgr0';stream.thread_count=2
            for i in range(4):
                pixels=np.full((24,32,3),i*40,np.uint8)
                frame=av.VideoFrame.from_ndarray(pixels,format='rgb24')
                for packet in stream.encode(frame):output.mux(packet)
            for packet in stream.encode():output.mux(packet)
        mapping=[];native=[];images=[]
        with av.open(str(source)) as source_video:
            source_video.streams.video[0].thread_count=2
            for i,frame in enumerate(source_video.decode(video=0)):
                pts=float(frame.pts*frame.time_base)
                mapping.append(dict(frame=i,pts_seconds=pts))
                native.append(dict(frame=i,pts_seconds=pts,native_pts=frame.pts,
                                   time_base_numerator=frame.time_base.numerator,time_base_denominator=frame.time_base.denominator))
                png=self.folder/f'frame_{i}.png';Image.fromarray(frame.to_ndarray(format='rgb24')).save(png)
                images.append(dict(frame=i,pts_seconds=pts,path=png.name,file_sha256=evaluator.sha(png)))
        self.record=dict(input_root=str(self.folder),input=dict(source_path=str(source),source_frame_pts=mapping,input_images=images))
        self.audit=dict(full_native_mapping=native)
        self.contact=dict(status='observed',frame=1,pts_seconds=mapping[1]['pts_seconds'])

    def tearDown(self):self.temp.cleanup()

    def test_valid_native_source(self):
        result=evaluator.native_check(self.record,self.audit,self.contact)
        self.assertTrue(result['all_native_pts_independently_verified'])
        self.assertTrue(result['selected_contact_png_rgb_verified'])

    def test_native_integer_tamper(self):
        changed=copy.deepcopy(self.audit);changed['full_native_mapping'][2]['native_pts']+=1
        with self.assertRaisesRegex(ValueError,'re-decode mismatch'):
            evaluator.native_check(self.record,changed,self.contact)

    def test_contact_png_tamper(self):
        Image.new('RGB',(32,24),'red').save(self.folder/'frame_1.png')
        with self.assertRaisesRegex(ValueError,'PNG differs'):
            evaluator.native_check(self.record,self.audit,self.contact)

    def test_contact_time_tamper(self):
        changed={**self.contact,'pts_seconds':self.contact['pts_seconds']+.01}
        with self.assertRaisesRegex(ValueError,'timestamp mismatch'):
            evaluator.native_check(self.record,self.audit,changed)


if __name__=='__main__':unittest.main()
