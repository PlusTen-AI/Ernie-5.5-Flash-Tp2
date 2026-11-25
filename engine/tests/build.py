#!/usr/bin/env python3
"""build.py — compile the engine module (every source under engine/src) into the extension cache; errors → log."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext  # noqa: E402

log = sys.argv[1] if len(sys.argv) > 1 else "/work/cache/build-engine.log"
t = time.time()
try:
    ext.load("engine", ext.engine_sources(), log=log, ldflags=ext.ENGINE_LDFLAGS)
except Exception as e:  # noqa: BLE001
    print(f"[build] FAILED after {time.time() - t:.0f} s: {str(e).splitlines()[-1] if str(e) else e} · log {log}")
    sys.exit(1)
print(f"[build] engine OK · {time.time() - t:.0f} s")
