"""Synthetic TEST_ONLY tempfile fixtures; never writes actual reviews/run bindings.

The only numeric ID used is a schema fixture, not a real acquired video or GT.
Actual case/media/model directories are not read. No GPU/model calls.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('test_only_review_importer',Path(__file__).with_name('import_reviews.py'))
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
import av
import numpy as np
from PIL import Image


class Contracts(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='TEST_ONLY_REVIEW_SCHEMA_');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.dist=self.root/'TEST_ONLY_DIST';self.case_dir=self.dist/'case';self.case_dir.mkdir(parents=True)
        source=self.root/'TEST_ONLY_NOT_NEXAR.mkv'
        with av.open(str(source),mode='w') as container:
            stream=container.add_stream('ffv1',rate=24);stream.width=32;stream.height=24;stream.pix_fmt='bgr0'
            for i in range(3):
                frame=av.VideoFrame.from_ndarray(np.full((24,32,3),(i*50,20,90),np.uint8),format='rgb24')
                for packet in stream.encode(frame):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        ui=[];times=[];inputs=[]
        with av.open(str(source)) as container:
            for i,frame in enumerate(container.decode(video=0)):
                path=self.case_dir/f'frame_{i:06d}.png';frame.to_image().convert('RGB').save(path)
                seconds=float(frame.pts*frame.time_base);relative=path.relative_to(self.dist).as_posix()
                ui.append(dict(frame=i,pts_seconds=seconds,image=relative,sha256=tool.sha(path)))
                times.append(dict(frame=i,pts_seconds=seconds))
                inputs.append(dict(frame=i,pts_seconds=seconds,path=relative,file_sha256=tool.sha(path)))
        cp=self.case_dir/'case.json';mp=self.case_dir/'mapping.json'
        case=dict(ID='NEXAR_REVIEW_00008',source_group_id='TEST_ONLY_SOURCE_GROUP',video_sha256=tool.sha(source),
                  source_uri='TEST_ONLY_NO_REMOTE_SOURCE',time_origin='TEST_ONLY_NATIVE_PTS',frames=ui)
        cp.write_text(json.dumps(case),encoding='utf-8')
        mp.write_text(json.dumps(dict(frame_pts=times,source_video_sha256=tool.sha(source))),encoding='utf-8')
        self.record=dict(ID='00008',case_path=str(cp),case_sha256=tool.sha(cp),mapping_path=str(mp),mapping_sha256=tool.sha(mp),input_root=str(self.dist),
                         input=dict(source_path=str(source),source_sha256=tool.sha(source),source_frame_pts=times,input_images=inputs,time_origin='TEST_ONLY_NATIVE_PTS'))
        self.review=dict(ID='NEXAR_REVIEW_00008',record_type='human_review_draft',evaluation_eligible=False,
                         source_video_sha256=tool.sha(source),source_group_id='TEST_ONLY_SOURCE_GROUP',source_uri='TEST_ONLY_NO_REMOTE_SOURCE',
                         time_origin='TEST_ONLY_NATIVE_PTS',frame_mapping_sha256=tool.js_frames_sha(cp),
                         review=dict(annotator='TEST_ONLY_FIXTURE_NOT_HUMAN',annotation_blinded=True,
                                     contact=dict(status='observed',frame=1,pts_seconds=times[1]['pts_seconds']),side='',entry=None,space=''))
        self.path=self.root/'TEST_ONLY_SYNTHETIC_REVIEW.json'

    def inspect(self):
        self.path.write_text(json.dumps(self.review),encoding='utf-8')
        return tool.inspect_review(self.path,self.record)

    def test_synthetic_native_and_pixels_metadata_only(self):
        result=self.inspect()
        self.assertTrue(result['eligible_for_contact_binding'])
        self.assertEqual(len(result['full_native_mapping']),3)
        self.assertFalse(result['ground_truth_promotion'])
        self.assertFalse((self.root/'review_binding.json').exists())

    def test_wrong_ID_and_source_SHA(self):
        self.review['ID']='NEXAR_REVIEW_99999';self.review['source_video_sha256']='0'*64
        result=self.inspect()
        self.assertIn('review_ID_matches_fixed_case',result['errors'])
        self.assertIn('source_SHA_all_equal',result['errors'])
        with self.assertRaises(ValueError):tool.select_reviews([self.path,self.path,self.path])

    def test_wrong_JS_mapping_hash(self):
        self.review['frame_mapping_sha256']='0'*64
        self.assertIn('full_JS_case_frames_hash',self.inspect()['errors'])

    def test_wrong_selected_PTS(self):
        self.review['review']['contact']['pts_seconds']+=.001
        self.assertIn('contact_native_PTS_exact',self.inspect()['errors'])

    def test_bool_frame_not_integer(self):
        self.review['review']['contact']['frame']=True
        self.assertIn('observed_contact_frame_valid',self.inspect()['errors'])

    def test_pixel_tamper_even_with_updated_metadata_hash(self):
        image=self.case_dir/'frame_000001.png';Image.new('RGB',(32,24),(1,2,3)).save(image)
        cp=Path(self.record['case_path']);case=json.loads(cp.read_text());case['frames'][1]['sha256']=tool.sha(image)
        cp.write_text(json.dumps(case),encoding='utf-8');self.record['case_sha256']=tool.sha(cp)
        self.record['input']['input_images'][1]['file_sha256']=tool.sha(image)
        self.review['frame_mapping_sha256']=tool.js_frames_sha(cp)
        result=self.inspect()
        self.assertTrue(result['checks']['PNG_1_SHA'])
        self.assertIn('selected_contact_PNG_pixels_equal',result['errors'])

    def test_uncertain_is_preserved_not_bound(self):
        self.review['review']['contact']=dict(status='uncertain',frame=None,pts_seconds=None)
        result=self.inspect()
        self.assertTrue(result['integrity_passed'])
        self.assertFalse(result['eligible_for_contact_binding'])
        self.assertEqual(result['contact_status'],'uncertain')
        self.assertIsNone(result['selected_contact_native'])
        self.assertFalse((self.root/'review_binding.json').exists())

    def test_preflight_refuses_overwrite_and_prediction_artifacts(self):
        run=self.root/'TEST_ONLY_RUN';run.mkdir();(run/'freeze.json').write_text('{}')
        output=self.root/'TEST_ONLY_OUTPUT'
        tool.preflight(run,output)
        output.mkdir()
        with self.assertRaises(ValueError):tool.preflight(run,output)
        for name in ('report.json','traces','review_binding.json'):
            marker=run/name
            marker.write_text('TEST_ONLY_EXISTENCE_MARKER_NOT_A_REVIEW_OR_BINDING')
            with self.assertRaises(ValueError):tool.preflight(run,self.root/'TEST_ONLY_NEW')
            marker.unlink()


if __name__=='__main__':unittest.main()
