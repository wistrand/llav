#!/usr/bin/env python3
"""Write the committed openapi.json from src/llav/openapi.py.

    PYTHONPATH=src python3 scripts/write-openapi.py

Run it whenever the API changes; tests/test_llav.py fails while the file is stale.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llav.openapi import document  # noqa: E402 - after the path is set

target = Path(__file__).resolve().parent.parent / "openapi.json"
target.write_text(json.dumps(document(), indent=2, sort_keys=True) + "\n")
print(f"Wrote {target}")
