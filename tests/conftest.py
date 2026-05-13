import sys
from pathlib import Path

# Make the project root importable in tests so `from app...` works.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
