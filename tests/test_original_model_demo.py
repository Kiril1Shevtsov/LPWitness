"""Frozen licensed real-issue artifacts; no optional solver dependency required."""
import importlib.util
from pathlib import Path
import unittest
from lpwitness import Problem, verify
import json
import shutil, subprocess, sys, tempfile
from fractions import Fraction

ROOT=Path(__file__).resolve().parents[1]/'examples/original_model_gate'
spec=importlib.util.spec_from_file_location('original_model_demo_verify',ROOT/'verify_demo.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class OriginalModelDemoTests(unittest.TestCase):
    def test_two_contracts_replay_and_negative_controls(self):
        records=module.check(ROOT/'results/final')
        self.assertEqual(len(records),2)
        self.assertTrue(all(r['reconstructed_verified'] for r in records))

    def test_decimal_model_is_frozen_source(self):
        p=Problem.from_dict(json.loads((ROOT/'models/original_decimal.json').read_text()))
        self.assertEqual(p.sha256,'06d79034280574cf211a941e5e0712f1feb5e3d290e5e4ba94b938e7405ca49f')
        self.assertEqual((len(p.c),len(p.E),len(p.G)),(33,32,15))
        self.assertEqual(sum(0<abs(x)<Fraction('1e-9') for row in p.E for x in row),20)

    def test_optimized_python_still_rejects_corrupted_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'artifacts'
            shutil.copytree(ROOT/'results/final',out)
            target=out/'source_decimal/reconstructed.certificate.json'
            proof=json.loads(target.read_text());proof['x'][0]='0'
            target.write_text(json.dumps(proof))
            result=subprocess.run([sys.executable,'-S','-O','-m',
                'examples.original_model_gate.verify_demo',str(out)],
                cwd=ROOT.parents[1],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('ValueError:',result.stderr)
            self.assertIn('primal_equalities',result.stderr)

    def test_swapped_contract_folder_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'artifacts'
            shutil.copytree(ROOT/'results/final',out)
            # An internally valid proof/model pair, but the wrong contract label.
            for file in (out/'source_decimal').iterdir():
                if file.is_file(): shutil.copy2(file,out/'solver_binary64'/file.name)
            with self.assertRaisesRegex(ValueError,'Wrong frozen model'):
                module.check(out)

    def test_direct_input_and_exact_model_must_match(self):
        spec=importlib.util.spec_from_file_location('original_model_demo_run',ROOT/'run_demo.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        p=Problem.from_dict(json.loads((ROOT/'models/original_decimal.json').read_text()))
        data=json.loads((ROOT/'models/linprog_input.json').read_text())
        runner.validate_inputs(p,data)
        data['c'][0]='1'
        with self.assertRaisesRegex(ValueError,'does not represent'):
            runner.validate_inputs(p,data)

    def test_contracts_are_not_interchangeable(self):
        binary=ROOT/'results/final/solver_binary64'
        decimal=ROOT/'results/final/source_decimal'
        p=Problem.from_dict(json.loads((binary/'model.json').read_text()))
        self.assertEqual(p.sha256,'9ad2a3f3edc199bb23f00795c5dd304b5d97538af7b5e1e4bfc44596a4703a4c')
        proof=json.loads((decimal/'reconstructed.certificate.json').read_text())
        self.assertFalse(verify(p,proof)['verified'])

if __name__=='__main__':unittest.main()
