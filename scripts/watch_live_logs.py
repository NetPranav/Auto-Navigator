"""
Real-time Live Log Monitor for magnum.log.
Watches magnum.log for new incoming entries, reports task plans, actions,
perceptions, and stops when 'stop' or session termination is detected.
"""

import sys
import time
from pathlib import Path

log_path = Path("magnum.log")

if not log_path.exists():
    log_path.touch()

print("🔍 Live log monitor active on magnum.log. Waiting for new activity...")
sys.stdout.flush()

# Start at end of file
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    f.seek(0, 2)  # Seek to end
    
    last_line = ""
    repeat_count = 0

    while True:
        line = f.readline()
        if not line:
            time.sleep(0.3)
            continue
        
        trimmed = line.strip()
        if not trimmed:
            continue
            
        # Deduplicate identical consecutive lines (e.g. repeated logs)
        if trimmed == last_line:
            repeat_count += 1
            if repeat_count == 3:
                print(f"  [... repeating previous line ...]")
                sys.stdout.flush()
            continue
        else:
            repeat_count = 0
            last_line = trimmed

        print(trimmed)
        sys.stdout.flush()

        lower = trimmed.lower()
        if "new user instruction" in lower:
            print(f"\n⚡ DETECTED NEW INSTRUCTION: {trimmed}\n")
            sys.stdout.flush()

        if "task result: completed" in lower or "goal completed" in lower:
            print(f"\n🎉 TASK COMPLETED: {trimmed}\n")
            sys.stdout.flush()

        if "stop" in lower and ("stopped" in lower or "cancelling" in lower or "cancelled" in lower):
            print(f"\n⏹️ STOP EVENT DETECTED: {trimmed}\n")
            sys.stdout.flush()
