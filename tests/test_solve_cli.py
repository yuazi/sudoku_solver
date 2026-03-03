import os
import subprocess
import sys
import unittest


class SolveCliTests(unittest.TestCase):
    def test_cli_rejects_non_digit_input_before_loading_model(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        puzzle = "a" + "0" * 80
        proc = subprocess.run(
            [sys.executable, "solve.py", puzzle],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("puzzle must contain only digits 0-9", proc.stdout)


if __name__ == "__main__":
    unittest.main()
