"""Authorization boundary tests; never execute a training runtime."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


LAUNCHER = Path(__file__).resolve().parents[2] / 'visual_bc/run_supervised_training_v2.py'
spec = importlib.util.spec_from_file_location('supervised_launcher_v2', LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class TrainingAuthorizationTests(unittest.TestCase):
    def test_audit_allows_unauthorized_protocol(self):
        with patch.object(launcher, 'validate_protocol', return_value=(
                {'training_authorized': False}, {'preflight_passed': True})), \
                patch.object(launcher.subprocess, 'Popen') as spawn, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run(Path('protocol.json'), 'pin'), 0)
            spawn.assert_not_called()

    def test_only_boolean_true_can_cross_training_gate(self):
        # Missing, false, and truthy non-booleans must all refuse before writes.
        for protocol in ({}, *({'training_authorized': value}
                              for value in (False, None, 0, 1, 'true', 'false', [], {}))):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as scratch:
                output = Path(scratch) / 'absent_training_output'
                with patch.object(launcher, 'validate_protocol', return_value=(protocol, {})), \
                        patch.object(launcher.subprocess, 'Popen') as spawn, \
                        patch.object(launcher, '_pinned_bytes') as reread, \
                        patch.object(launcher.Path, 'mkdir') as mkdir, \
                        patch.object(launcher, '_write_json_new') as write:
                    with self.assertRaisesRegex(ValueError, 'Full persistent supervised training is not authorized'):
                        launcher.run(Path('protocol.json'), 'pin', train=True, output_directory=output)
                    for operation in (spawn, reread, mkdir, write):
                        operation.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
