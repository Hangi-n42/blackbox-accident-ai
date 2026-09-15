"""No torch/weights/network; synthetic source manifests and gate schemas only."""
import sys,json,copy,tempfile,unittest,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nf4_8b_io_guards',HERE/'nf4_8b_io.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Guards(unittest.TestCase):
    def setUp(self):
        self.plan={'repository':m.REPO,'revision':m.REV,'license':'apache-2.0','files':[{'name':'part.safetensors','bytes':12,'url':f'https://huggingface.co/{m.REPO}/resolve/{m.REV}/part.safetensors','lfs_sha256':'a'*64}]}
        self.download={'repository':m.REPO,'revision':m.REV,'status':'complete','files':[{'name':'part.safetensors','bytes':12,'sha256':'a'*64,'expected_lfs_sha256':'a'*64}]}
    def test_pinned_identity_valid_schema(self):m.validate_identity(self.plan,self.download)
    def test_wrong_repository_revision_incomplete_are_rejected(self):
        for k,v in [('repository','other/model'),('revision','main'),('status','running')]:
            d=copy.deepcopy(self.download);d[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):m.validate_identity(self.plan,d)
    def test_shard_mismatch_duplicate_missing_rejected(self):
        for variant in ('hash','duplicate','missing','size'):
            d=copy.deepcopy(self.download)
            if variant=='hash':d['files'][0]['sha256']='b'*64
            if variant=='duplicate':d['files']*=2
            if variant=='missing':d['files']=[]
            if variant=='size':d['files'][0]['bytes']=13
            with self.subTest(variant=variant),self.assertRaises(ValueError):m.validate_identity(self.plan,d)
    def test_unpinned_url(self):
        p=copy.deepcopy(self.plan);p['files'][0]['url']=p['files'][0]['url'].replace(m.REV,'main')
        with self.assertRaises(ValueError):m.validate_identity(p,self.download)
    def test_safe_members(self):
        for name in ('../escape','C:/escape','/escape','sub\\escape',''):
            with self.subTest(name=name),self.assertRaises(ValueError):m.member(HERE,name)
        self.assertEqual(m.member(HERE,'valid/config.json'),(HERE/'valid/config.json').resolve())
    def test_quantization_exact_and_wrong_precision(self):
        q={'quant_method':'bitsandbytes','load_in_4bit':True,'bnb_4bit_quant_type':'nf4','bnb_4bit_compute_dtype':'float16','bnb_4bit_use_double_quant':False};m.validate_quant(q)
        for k,v in [('bnb_4bit_quant_type','fp4'),('bnb_4bit_compute_dtype','bfloat16'),('bnb_4bit_use_double_quant',True),('load_in_4bit',False)]:
            bad=dict(q,**{k:v})
            with self.subTest(k=k),self.assertRaises(ValueError):m.validate_quant(bad)
    def test_architecture_and_tied_rejected(self):
        c={'model_type':'qwen3_vl','tie_word_embeddings':False,'text_config':{'vocab_size':151936,'hidden_size':4096}};m.validate_config(c)
        c['tie_word_embeddings']=True
        with self.assertRaises(ValueError):m.validate_config(c)
    def test_missing_gate_no_output_or_overwrite(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp);target=root/'new';m.new_output(target)
            with self.assertRaises(FileNotFoundError):m.verify_development(root,root)
            self.assertFalse(target.exists())
            with self.assertRaises(ValueError):m.new_output(root)
    def test_synthetic_development_numeric_gate(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();m.put(source/'download_manifest.json',{'synthetic':True})
            report={'status':'complete','runtime':{'precision':'nf4','compute_dtype':'float16','double_quant':False},'videos':[{'ID':ID,'baseline':{'collision_frame':12},'candidate':{'collision_frame':12}} for ID in ('00000','00003','00004','00005','00006','00007','00008','00010','00013')]}
            m.put(root/'report.json',report);m.put(root/'freeze.json',{'files':{str(source/'download_manifest.json'):m.sha(source/'download_manifest.json')}})
            s={'contact':{'n':9,'accuracy':.2},'entry':{'n':5,'accuracy':.2},'side':{'n':4,'macro_f1':.5},'space':{'n':6,'macro_f1':.5}}
            ev={'report_sha256':m.sha(root/'report.json'),'scores':{'baseline':s,'candidate':copy.deepcopy(s)}};ev['scores']['candidate']['entry']['accuracy']=.4
            p=root/'evaluation.json';m.put(p,ev);m.verify_development(root,source)
            for field,key,value in [('entry','accuracy',.1),('side','macro_f1',.4),('space','n',0),('contact','accuracy',.3),('entry','accuracy',float('nan'))]:
                bad=copy.deepcopy(ev);bad['scores']['candidate'][field][key]=value;p.write_text(json.dumps(bad))
                with self.subTest(field=field,key=key),self.assertRaises(ValueError):m.verify_development(root,source)
            ev['scores']['candidate']=copy.deepcopy(s);p.write_text(json.dumps(ev))
            with self.assertRaises(ValueError):m.verify_development(root,source)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    m.put(HERE/'nf4_8b_io_cpu_guard_results.json',{'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
        'failures':[(str(t),e) for t,e in result.failures+result.errors],'helper_sha256':m.sha(HERE/'nf4_8b_io.py'),'test_sha256':m.sha(Path(__file__)),
        'torch_imported':'torch' in sys.modules,'GPU_or_model_execution':False,'real_weights_read':False,'export_created':False,
        'scope':'Synthetic schema and rejection contracts. Does not validate GPU export or reload behavior.'})
    raise SystemExit(not result.wasSuccessful())
