"""CPU pixel contracts for actual parent mosaic -> exact tiles; no VLM/model."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key]='2'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def load(name):
    full='solution.'+name
    spec=importlib.util.spec_from_file_location(full,Path(__file__).with_name(name+'.py'))
    mod=importlib.util.module_from_spec(spec);sys.modules[full]=mod;spec.loader.exec_module(mod)
    return mod


parent=load('stage2_contact_verify_v6a')
candidate=load('stage2_contact_tiles_v6b')
from PIL import Image


class FakeVLM:
    def __init__(self, answer='null'):
        self.answer=answer;self.calls=[];self.pixel_budget=1200000

    def ask(self,images,prompt,max_new_tokens):
        self.calls.append((images,prompt,max_new_tokens))
        return self.answer


class Contracts(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        folder=Path(temp.name);self.paths=[]
        for i in range(60):
            path=folder/f'frame_{11+7*i:06d}.png'
            image=Image.new('RGB',(47+i%4,31+i%3),(i*3,i*2,i))
            image.putpixel((i%image.width,i%image.height),(255,0,255))
            image.save(path);self.paths.append(path)
        self.pred=dict(collision_frame=326,entry_frame=11,entry_side='RIGHT',evasion_space=1)
        self.diag={'collision_replacement':{'base_collision_frame':32}}

    def compare(self,paths=None,pred=None,diag=None,answer='null'):
        paths=paths or self.paths;pred=pred or self.pred;diag=diag or self.diag
        original=copy.deepcopy((pred,diag))
        a,b=FakeVLM(answer),FakeVLM(answer)
        pa,da=parent.refine_collision(paths,pred,diag,a)
        pb,db=candidate.refine_collision(paths,pred,diag,b)
        self.assertEqual(pa,pb)
        self.assertEqual(da['contact_verification'],db['contact_verification'])
        self.assertEqual((pred,diag),original)
        self.assertEqual(len(a.calls),1);self.assertEqual(len(b.calls),1)
        mosaic=a.calls[0][0][0];tiles,prompt,tokens=b.calls[0]
        self.assertEqual(prompt,a.calls[0][1]);self.assertEqual(tokens,a.calls[0][2]);self.assertEqual(tokens,48)
        count=len(da['contact_verification']['offered_original_frames'])
        self.assertEqual(len(tiles),count);self.assertLessEqual(count,18)
        columns=min(5,count)
        for i,tile in enumerate(tiles):
            expected=mosaic.crop((i%columns*384,i//columns*256,(i%columns+1)*384,(i//columns+1)*256))
            self.assertEqual(tile.size,(384,256));self.assertEqual(tile.tobytes(),expected.tobytes())
            self.assertEqual(db['presentation']['tile_rgb_sha256'][i],hashlib.sha256(expected.convert('RGB').tobytes()).hexdigest())
        self.assertEqual(b.pixel_budget,1200000)
        return pb,db

    def test_actual_mosaic_order_pixels_and_no_padding(self):
        _,detail=self.compare()
        self.assertEqual(detail['presentation']['tile_count'],18)
        self.assertEqual(detail['presentation']['mosaic_size'],[1920,1024])

    def test_one_to_four_tiles_column_clamp(self):
        for n in range(1,5):
            pred={**self.pred,'collision_frame':11+7*(n-1)}
            diag={'collision_replacement':{'base_collision_frame':11}}
            _,d=self.compare(self.paths[:n],pred,diag)
            self.assertEqual(d['presentation']['tile_count'],n)
            self.assertEqual(d['presentation']['columns'],n)

    def test_overlap_centers_omit_padding_and_duplicate_tiles(self):
        diag={'collision_replacement':{'base_collision_frame':self.pred['collision_frame']}}
        _,d=self.compare(diag=diag)
        self.assertEqual(d['presentation']['tile_count'],9)

    def test_valid_and_invalid_responses_parent_parser_unchanged(self):
        for raw in ['{"collision_frame":32}','{"collision_frame":true}','{"collision_frame":"32"}','{"collision_frame":32.0}','{"collision_frame":999999}','invalid']:
            with self.subTest(raw=raw):
                result,_=self.compare(answer=raw)
                for k in ('entry_frame','entry_side','evasion_space'):
                    self.assertEqual(result[k],self.pred[k])

    def test_bad_mosaic_and_repeated_call_rejected(self):
        fake=FakeVLM();adapter=candidate._TileAdapter(fake,6)
        with self.assertRaises(ValueError):adapter.ask([Image.new('RGB',(384,256))],'prompt',48)
        adapter.ask([Image.new('RGB',(1920,512))],'prompt',48)
        with self.assertRaises(ValueError):adapter.ask([Image.new('RGB',(1920,512))],'prompt',48)
        self.assertEqual(len(fake.calls),1)

    def test_baseline_called_once_then_tile_verifier(self):
        fake=FakeVLM()
        with patch.object(candidate.baseline,'_predict_file',return_value=(self.pred,self.diag)) as old:
            result,diag=candidate._predict_file(self.paths,'SCORES',fake)
        old.assert_called_once_with(self.paths,'SCORES',fake)
        self.assertEqual(len(fake.calls),1)
        self.assertEqual(result,self.pred)
        self.assertIn('presentation',diag)


if __name__=='__main__':unittest.main()
