"""In-memory metadata contracts; fixture bytes are NOT acquired footage."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from check_capture_manifest import audit


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        raw = b'UNIT TEST ONLY; NOT VIDEO'
        (self.base/'fixture.bin').write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        ref = {'path': 'fixture.bin', 'sha256': digest}
        common = {'source_group':'source-A', 'content_id':'content-A', 'original_camera_id':'original-camera-A',
                  'domain':'road', 'status':'human_verified', 'split':'holdout', **ref,
                  'media':{'codec':'test-only', 'width':1, 'height':1, 'decoded_frame_count':1, 'timing_basis':'fixture'},
                  'rights':{'status':'verified','basis':'fixture','permitted_uses':['competition_evaluation']},
                  'annotation':{'reviewer':'fixture','reviewed_at':'fixture','label_basis':'fixture','model_predictions_seen':False},
                  'capture_conditions':{'notes':'fixture only'}}
        original = {**copy.deepcopy(common), 'record_id':'original-A', 'acquisition_kind':'original_camera','label':'ORIGINAL',
                    'parent_original_id':None,'display_id':None,'recapture_camera_id':None,
                    'evidence':[{**ref,'kind':'source_provenance'}]}
        recapture = {**copy.deepcopy(common), 'record_id':'recapture-A','acquisition_kind':'physical_display_recapture','label':'RERECORDED',
                     'parent_original_id':'original-A','display_id':'display-A','recapture_camera_id':'camera-A',
                     'evidence':[{**ref,'kind':'setup_photo_or_video'},{**ref,'kind':'capture_log'}]}
        other = raw + b'RECAPTURE FIXTURE'
        (self.base/'recapture.bin').write_bytes(other)
        recapture.update(path='recapture.bin', sha256=hashlib.sha256(other).hexdigest())
        self.data = {'schema_version':1, 'protocol':{'frozen_at':'fixture','selection_rule':'fixture','required_use':'competition_evaluation',
                                                    'device_holdout_fields':['display_id','recapture_camera_id','original_camera_id']},
                     'records':[original,recapture]}

    def test_metadata_contract_not_media_proof(self):
        # This deliberately passes non-video bytes: checks cannot certify truthful evidence.
        self.assertEqual(audit(self.data,self.base)['status'],'METADATA_CHECKS_PASS')

    def test_empty_not_ready(self):
        self.data['records']=[]
        self.assertEqual(audit(self.data,self.base)['status'],'NOT_READY')

    def test_synthetic_not_positive(self):
        self.data['records'][1]['acquisition_kind']='synthetic_transform'
        self.assertEqual(audit(self.data,self.base)['provisional_row_eligible_counts']['road_physical_recaptures'],0)

    def test_source_and_device_overlap(self):
        extra=copy.deepcopy(self.data['records'][0]); extra['record_id']='original-B'; extra['split']='development'
        self.data['records'].append(extra)
        errors=audit(self.data,self.base)['errors']
        self.assertTrue(any('source_group' in e for e in errors))
        self.assertTrue(any('original_camera_id' in e for e in errors))

    def test_missing_evidence_and_wrong_sha(self):
        self.data['records'][1]['evidence']=[]
        self.data['records'][0]['sha256']='0'*64
        errors=audit(self.data,self.base)['errors']
        self.assertTrue(any('physical capture needs' in e for e in errors))
        self.assertTrue(any('SHA mismatch' in e for e in errors))

    def test_unreviewed_or_unlicensed_not_ready(self):
        self.data['records'][1]['status']='acquired'
        self.data['records'][0]['rights']['status']='pending'
        self.assertEqual(audit(self.data,self.base)['status'],'NOT_READY')

    def test_parent_mismatch(self):
        self.data['records'][1]['content_id']='other-content'
        self.assertTrue(any('parent mismatch' in e for e in audit(self.data,self.base)['errors']))

    def test_metadata_only_never_ready(self):
        self.assertEqual(audit(self.data,self.base,False)['status'],'NOT_READY')

    def test_same_asset_conflicting_labels(self):
        self.data['records'][1]['path']=self.data['records'][0]['path']
        self.data['records'][1]['sha256']=self.data['records'][0]['sha256']
        self.assertTrue(any('conflicting class labels' in e for e in audit(self.data,self.base)['errors']))


if __name__ == '__main__':
    unittest.main()
