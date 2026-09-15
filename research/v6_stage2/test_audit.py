"""Synthetic contracts only. These results are not measured model accuracy."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
import sys

import audit


class AuditContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audit_contract_", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("source", "mapping", "official", "review_A", "review_B", "adjudication"):
            (self.root / name).write_text(name, encoding="utf-8")
        self.prov = dict(artifact_path="official", sha256=audit.sha(self.root / "official"),
            source_ref="Synthetic contract fixture, NOT real official labels", created_at="2026-09-14T00:00:00+00:00",
            annotators=[], adjudicator=None)
        self.numbers = [10, 20, 40, 80]
        self.pts = [0.0, 0.1, 1.4, 2.6]

    def label(self, value):
        return dict(value=value, gt_type="official", provenance=copy.deepcopy(self.prov))

    def fixture(self, internal=80, offered=None, entry=40):
        offered = offered or [10, 40, 80]
        video = dict(ID="contract", source_group_id="synthetic_source", source=dict(dataset="contract_fixture",
            source_uri="synthetic-not-a-dataset", video_path="source", video_sha256=audit.sha(self.root / "source")),
            split="development", exposure=dict(predictions_seen=True, prior_model_development=True,
                annotation_blinded=False, notes="Invented numbers only test software contracts, not performance"),
            target_vehicle_id="synthetic_target", input_manifest_sha256="0" * 64,
            frame_pts=[dict(frame=n, pts_seconds=t) for n,t in zip(self.numbers,self.pts)],
            mapping_provenance=dict(artifact_path="mapping", sha256=audit.sha(self.root / "mapping"),
                method="synthetic variable-duration frame mapping", time_origin="synthetic PTS origin"),
            labels=dict(collision_time_seconds=self.label(1.4), entry_time_seconds=self.label(1.4),
                entry_side=self.label("LEFT"), evasion_space=self.label(0)))
        diagnostics = dict(collision_replacement=dict(base_collision_frame=internal, collision_frame=40),
            entry_candidates=offered, collision_candidates=[10,20,40,80], entry=dict(entry_frame=entry),
            coarse=dict(entry_side="LEFT"), space=dict(evasion_space=0))
        row = dict(ID="contract", input_manifest_sha256="0"*64, frame_numbers=self.numbers,
            prediction=dict(collision_frame=40, entry_frame=entry, entry_side="LEFT", evasion_space=0),
            diagnostics=diagnostics, calls=[dict(status="complete", raw_output="{}") for _ in range(4)])
        self.bind_mapping(video)
        return dict(schema_version=1,cohort_id="synthetic",videos=[video]), dict(schema_version=1,
            scope="synthetic_contract_only", videos=[audit.normalize_trace(row)])

    def bind_mapping(self, video):
        (self.root/"mapping").write_text(json.dumps(dict(frame_pts=video["frame_pts"],
            source_video_sha256=video["source"]["video_sha256"],time_origin=video["mapping_provenance"]["time_origin"])))
        video["mapping_provenance"]["sha256"]=audit.sha(self.root/"mapping")

    def run_fixture(self, **kwargs):
        gt, trace = self.fixture(**kwargs)
        return audit.evaluate(gt,trace,self.root)

    def test_noncontiguous_numbers_use_pts_not_number_or_fps(self):
        result = self.run_fixture(internal=20, offered=[10,20], entry=20)
        errors = result["videos"][0]["collision_errors"]
        self.assertEqual(errors["final_motion"]["signed_seconds"], 0.0)
        self.assertAlmostEqual(errors["internal_vlm"]["signed_seconds"], -1.3)

    def test_upper_bound_loss_distinct_from_grid_and_selection(self):
        bound = self.run_fixture(internal=20, offered=[10,20], entry=20)
        grid = self.run_fixture(offered=[10,80], entry=10)
        selection = self.run_fixture(offered=[10,40,80], entry=10)
        reasons = [r["videos"][0]["entry_coverage"]["first_failure"] for r in (bound,grid,selection)]
        self.assertEqual(reasons,["internal_collision_upper_bound_loss","entry_grid_loss","selection_error"])

    def test_source_coverage_is_not_grid_loss(self):
        gt,trace = self.fixture()
        gt["videos"][0]["labels"]["entry_time_seconds"]["value"] = 0.7
        result=audit.evaluate(gt,trace,self.root)
        self.assertEqual(result["videos"][0]["entry_coverage"]["first_failure"],"original_time_coverage_missing")

    def test_missing_label_keeps_component_denominator_and_null_s2(self):
        gt,trace=self.fixture()
        gt["videos"][0]["labels"]["entry_side"]=None
        result=audit.evaluate(gt,trace,self.root)
        self.assertIsNone(result["S2"])
        self.assertEqual(result["components"]["entry_side"]["labeled_count"],0)
        self.assertEqual(result["components"]["collision_time_seconds"]["value"],1)

    def test_macro_uses_full_classes_and_formula(self):
        # One-class synthetic cohort: the absent defined class has zero F1.
        result=self.run_fixture()
        self.assertEqual(result["components"]["entry_side"]["value"],0.5)
        self.assertEqual(result["components"]["evasion_space"]["value"],0.5)
        self.assertAlmostEqual(result["S2"],0.85)
        mixed=audit.macro_f1(["LEFT","RIGHT","RIGHT"],["LEFT","LEFT","RIGHT"],("LEFT","RIGHT"))
        self.assertAlmostEqual(mixed["value"],2/3)

    def test_invalid_class_counts_wrong_without_creating_extra_class(self):
        gt,trace=self.fixture()
        trace["videos"][0]["prediction"]["entry_side"]="INVALID"
        result=audit.evaluate(gt,trace,self.root)
        self.assertEqual(result["components"]["entry_side"]["value"],0)

    def test_ai_and_single_human_draft_rejected(self):
        for kind in ("AI_advisory","human_draft","human_adjudicated"):
            gt,trace=self.fixture()
            gt["videos"][0]["labels"]["entry_side"]["gt_type"]=kind
            with self.assertRaises(ValueError):
                audit.evaluate(gt,trace,self.root)

    def test_two_human_review_artifacts_and_resolution_accepted(self):
        gt,trace=self.fixture()
        label=gt["videos"][0]["labels"]["entry_side"]
        label["gt_type"]="human_adjudicated"
        label["provenance"].update(annotators=["A","B"],adjudicator="A",
            independent_reviews=[dict(annotator=x, artifact_path=f"review_{x}",sha256=audit.sha(self.root/f"review_{x}")) for x in ("A","B")],
            adjudication=dict(artifact_path="adjudication",sha256=audit.sha(self.root/"adjudication"),resolution="Synthetic review agreement"))
        self.assertTrue(audit.evaluate(gt,trace,self.root)["complete_gt"])

    def test_exposed_data_cannot_claim_validation(self):
        gt,trace=self.fixture()
        gt["videos"][0]["split"]="validation"
        with self.assertRaises(ValueError):audit.evaluate(gt,trace,self.root)

    def test_missing_pts_and_hash_mismatch_rejected(self):
        for key in ("frame_pts","input_manifest_sha256"):
            gt,trace=self.fixture()
            gt["videos"][0][key]=[] if key=="frame_pts" else "1"*64
            with self.assertRaises(ValueError):audit.evaluate(gt,trace,self.root)

    def test_first_valid_and_default_are_not_semantic_already_inside(self):
        gt,trace=self.fixture(entry=10)
        valid=audit.evaluate(gt,trace,self.root)
        self.assertEqual(valid["videos"][0]["selection_provenance"]["entry"],"valid_first_selected_semantics_unverified")
        trace["videos"][0]["diagnostics"]["entry"]={}
        invalid=audit.evaluate(gt,trace,self.root)
        self.assertEqual(invalid["videos"][0]["selection_provenance"]["entry"],"invalid_fallback_first")

    def test_pts_origin_and_filename_offset_invariance(self):
        gt,trace=self.fixture()
        expected=audit.evaluate(gt,trace,self.root)
        for item in gt["videos"][0]["frame_pts"]:item["pts_seconds"]+=17
        for field in audit.FIELDS[:2]:gt["videos"][0]["labels"][field]["value"]+=17
        self.bind_mapping(gt["videos"][0])
        shifted=audit.evaluate(gt,trace,self.root)
        self.assertEqual(expected["S2"],shifted["S2"])
        self.assertTrue(audit.within(0.3))
        self.assertFalse(audit.within(0.30001))

    def test_mapping_content_cannot_be_changed_while_evidence_hash_stays_same(self):
        gt,trace=self.fixture()
        gt["videos"][0]["frame_pts"][2]["pts_seconds"]=1.5
        with self.assertRaises(ValueError):audit.evaluate(gt,trace,self.root)

    def test_original_filename_offset_preserves_all_time_results(self):
        gt,trace=self.fixture()
        expected=audit.evaluate(gt,trace,self.root)
        for item in gt["videos"][0]["frame_pts"]:item["frame"]+=700
        self.bind_mapping(gt["videos"][0])
        row=trace["videos"][0]
        for field in ("original_frame_numbers","valid_frame_numbers"):
            row[field]=[n+700 for n in row[field]]
        for field in ("collision_frame","entry_frame"):row["prediction"][field]+=700
        d=row["diagnostics"]
        for field in ("collision_candidates","entry_candidates"):d[field]=[n+700 for n in d[field]]
        for field in ("base_collision_frame","collision_frame"):d["collision_replacement"][field]+=700
        d["entry"]["entry_frame"]+=700
        actual=audit.evaluate(gt,trace,self.root)
        self.assertEqual(expected,actual)

    def test_decode_filter_loss_has_separate_bucket(self):
        gt,trace=self.fixture(entry=20,offered=[10,20,80])
        row=trace["videos"][0]
        row["valid_frame_numbers"]=[10,20,80]
        row["diagnostics"]["collision_candidates"]=[10,20,80]
        row["diagnostics"]["collision_replacement"]["collision_frame"]=20
        row["prediction"]["collision_frame"]=20
        result=audit.evaluate(gt,trace,self.root)
        self.assertEqual(result["videos"][0]["entry_coverage"]["first_failure"],"decode_filter_loss")

    def test_evaluation_does_not_mutate_source_or_share_video_state(self):
        gt,trace=self.fixture()
        before=json.dumps([gt,trace],sort_keys=True)
        first=audit.evaluate(gt,trace,self.root)
        self.run_fixture(internal=20,offered=[10,20],entry=20)
        self.assertEqual(first,audit.evaluate(gt,trace,self.root))
        self.assertEqual(before,json.dumps([gt,trace],sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--report":
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditContracts))
        audit.write_new(sys.argv[2],dict(created_utc=datetime.now(timezone.utc).isoformat(),
            passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            scope="Synthetic software contracts only; NOT model accuracy or real GT",
            evaluator_sha256=audit.sha(Path(__file__).parent/"audit.py"),tests_sha256=audit.sha(__file__)))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    unittest.main(verbosity=2)
