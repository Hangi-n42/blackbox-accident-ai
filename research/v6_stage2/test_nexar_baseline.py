"""Small CPU contracts for timestamp sampler and transparent ask recorder."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).with_name("run_nexar_baseline.py")
spec=importlib.util.spec_from_file_location("nexar_runner",path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class Contracts(unittest.TestCase):
    def test_irregular_pts_preserves_original_indices_and_endpoints(self):
        self.assertEqual(runner.sampled_indices([7.,7.04,7.11,7.205,7.31,7.315]),[0,2,3,4,5])

    def test_tie_earlier_and_duplicate_selected_indices_removed(self):
        self.assertEqual(runner.sampled_indices([0.,.05,.15,.25]),[0,1,2,3])
        self.assertEqual(runner.sampled_indices([0.,1.]),[0,1])

    def test_origin_shift_invariance_and_single_frame(self):
        times=[0.,.031,.099,.205,.241]
        self.assertEqual(runner.sampled_indices(times),runner.sampled_indices([x+13 for x in times]))
        self.assertEqual(runner.sampled_indices([4.7]),[0])

    def test_bad_pts_rejected(self):
        for times in ([],[0,0],[1,0],[0,float("nan")]):
            with self.assertRaises(ValueError):runner.sampled_indices(times)

    def test_only_declared_development_sources(self):
        self.assertEqual(runner.IDS,("00000","00003"))
        self.assertEqual(runner.POLICY["tokens"],[64,48,40,40])

    def test_recorder_is_transparent_and_per_file_budget(self):
        from PIL import Image
        class CUDA:
            synchronize=staticmethod(lambda:None)
            reset_peak_memory_stats=staticmethod(lambda:None)
            max_memory_allocated=staticmethod(lambda:0)
            max_memory_reserved=staticmethod(lambda:0)
        class Torch:cuda=CUDA()
        class Model:
            pixel_budget=1200000;torch=Torch()
            def __init__(self):self.received=[]
            def ask(self,images,prompt,max_new_tokens):
                self.received.append((images,prompt,max_new_tokens))
                return '{"collision_frame":42}'
        model=Model();image=Image.new("RGB",(64,64));images=[image]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            folder=Path(temp);ledger={"call_count":0}
            first=runner.Recorder(model,folder,ledger)
            prompt="Available frames: [7, 42]."
            for tokens in runner.POLICY["tokens"]:
                self.assertEqual(first.ask(images,prompt,tokens),'{"collision_frame":42}')
            self.assertIs(model.received[0][0],images)
            self.assertEqual(first.calls[0]["offered_original_numbers"],[7,42])
            with self.assertRaises(ValueError):first.ask(images,prompt,64)
            second_folder=folder/"second";second_folder.mkdir()
            second=runner.Recorder(model,second_folder,ledger)
            second.ask(images,prompt,64)
            self.assertEqual(len(second.calls),1)
            self.assertEqual(ledger["call_count"],5)


if __name__=="__main__":unittest.main(verbosity=2)
