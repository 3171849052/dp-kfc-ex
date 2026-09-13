import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def pytest_sessionfinish(session, exitstatus):
    (Path(__file__).parent/'results'/'correctness.json').write_text(json.dumps({
        'passed': exitstatus == 0, 'exit_status': int(exitstatus),
        'tests_collected': session.testscollected, 'tests_failed': session.testsfailed}, indent=2))
