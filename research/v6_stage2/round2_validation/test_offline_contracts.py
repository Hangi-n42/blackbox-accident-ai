"""Synthetic file-name/output tests; never touches real reviews or inference data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import pandas as pd

spec = importlib.util.spec_from_file_location('offline_contracts',Path(__file__).with_name('offline_contracts.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class Contracts(unittest.TestCase):
    def test_actual_driver_refuses_missing_accuracy_evidence(self):
        here=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_no_gate_',dir=here) as temp:
            root=Path(temp).resolve()
            self.assertTrue(root.is_relative_to(here))
            destination=root/'output'
            command=[sys.executable,'-I','-B',str(here/'verify_v6_offline.py')]
            for flag in ('package','manifest','validation-run','stage1-data','stage3-data','baseline-results'):
                command.extend(['--'+flag,str(root/flag)])
            command.extend(['--output',str(destination)])
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('evaluation.json',result.stderr)
            self.assertFalse(destination.exists())

    def test_png_jpeg_original_numbers(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_output_contract_') as temp:
            folder = Path(temp)/'TEST_ONLY'
            folder.mkdir()
            (folder/'frame_000007.png').touch()
            (folder/'frame_000010.JPG').touch()
            (folder/'case.json').write_text('{}')
            row = dict(ID='TEST_ONLY',collision_frame=7,entry_frame=10,evasion_space=0,entry_side='LEFT')
            frame = pd.DataFrame([row])
            c.validate_stage2(frame,temp)
            with self.assertRaises(ValueError):
                c.validate_stage2(frame.assign(collision_frame=8),temp)
            with self.assertRaises(ValueError):
                c.validate_stage2(frame.assign(evasion_space=False),temp)
            with self.assertRaises(ValueError):
                c.validate_stage2(frame.assign(entry_frame=10.0),temp)
            (folder/'other_000007.jpg').touch()
            with self.assertRaises(ValueError):
                c.validate_stage2(frame,temp)

    def test_malformed_image_number(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_output_contract_') as temp:
            (Path(temp)/'frame_000007.interrupted.png').touch()
            with self.assertRaises(ValueError):
                c.frame_numbers(temp)

    def test_actual_report_variant_and_output_equivalence(self):
        # These are in-memory prediction fixtures, not human labels or a validation report on disk.
        videos=[dict(ID=ID,candidate=dict(collision_frame=2,entry_frame=1,evasion_space=1,entry_side='RIGHT'))
                for ID in ('00008','00010','00013')]
        report=dict(status='complete',phase='round2_fresh_validation',call_count=12,model_loads=1,
                    network_attempts=0,videos=videos)
        result=pd.DataFrame([dict(ID=v['ID'],**v['candidate']) for v in videos])
        c.compare_expected_stage2(result,report)
        with self.assertRaises(AssertionError):
            c.compare_expected_stage2(result.assign(collision_frame=3),report)
        with self.assertRaises(ValueError):
            c.compare_expected_stage2(result,dict(report,phase='old_development'))
        with self.assertRaises(ValueError):
            c.compare_expected_stage2(result,dict(report,network_attempts=1))


if __name__ == '__main__':
    unittest.main()
