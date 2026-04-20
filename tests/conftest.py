import sys
from pathlib import Path

# Put the server source on the path so tests can import server modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "server"))