#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\]")
TOTAL_LOSS_RE = re.compile(r"\bTotal loss\b", re.IGNORECASE)
RUNNING_RE = re.compile(r"\bRunning experiment\b", re.IGNORECASE)
LOADED_DATA_RE = re.compile(r"\bLoaded training data with\b", re.IGNORECASE)

def parse_ts(line: str):
    m = TS_RE.match(line)
    if not m:
        return None
    ts = f"{m.group(1)},{m.group(2)}"
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")

def analyze_log(path: Path, warmup_steps: int = 0, start_anchor: str = "auto"):
    """
    start_anchor:
      - "auto": prefer first "Loaded training data" else first "Running experiment" else first Total loss
      - "running": first "Running experiment"
      - "loaded": first "Loaded training data"
      - "first_total": first "Total loss"
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    total_loss_times = []
    running_times = []
    loaded_times = []

    for line in lines:
        ts = parse_ts(line)
        if ts is None:
            continue
        if TOTAL_LOSS_RE.search(line):
            total_loss_times.append(ts)
        if RUNNING_RE.search(line):
            running_times.append(ts)
        if LOADED_DATA_RE.search(line):
            loaded_times.append(ts)

    # pick anchors
    def pick_start():
        if start_anchor == "running":
            return running_times[0] if running_times else None
        if start_anchor == "loaded":
            return loaded_times[0] if loaded_times else None
        if start_anchor == "first_total":
            return total_loss_times[0] if total_loss_times else None
        # auto
        if loaded_times:
            return loaded_times[0]
        if running_times:
            return running_times[0]
        if total_loss_times:
            return total_loss_times[0]
        return None

    start_time = pick_start()
    end_time = total_loss_times[-1] if total_loss_times else None

    # steps based on Total loss lines
    if warmup_steps > 0 and len(total_loss_times) > warmup_steps:
        tl = total_loss_times[warmup_steps:]
    else:
        tl = total_loss_times[:]

    # durations
    def seconds(a, b):
        return (b - a).total_seconds()

    # primary efficiency window: from first (post-warmup) Total loss to last Total loss
    if len(tl) >= 2:
        eff_start = tl[0]
        eff_end = tl[-1]
        eff_seconds = seconds(eff_start, eff_end)
        steps = len(tl)
        # There are (steps-1) intervals between timestamps
        sec_per_step = eff_seconds / (steps - 1) if steps > 1 else None
        steps_per_sec = (steps - 1) / eff_seconds if eff_seconds > 0 else None

        # also compute interval stats
        intervals = [seconds(tl[i], tl[i+1]) for i in range(len(tl)-1)]
        interval_mean = mean(intervals)
        interval_min = min(intervals)
        interval_max = max(intervals)
    else:
        eff_start = eff_end = None
        eff_seconds = 0.0
        steps = len(tl)
        sec_per_step = None
        steps_per_sec = None
        interval_mean = interval_min = interval_max = None

    # overall runtime window: from start_time to end_time
    overall_seconds = seconds(start_time, end_time) if (start_time and end_time) else None

    return {
        "log_path": str(path),
        "anchors": {
            "start_anchor_mode": start_anchor,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time_last_total_loss": end_time.isoformat() if end_time else None,
            "first_total_loss": total_loss_times[0].isoformat() if total_loss_times else None,
            "last_total_loss": total_loss_times[-1].isoformat() if total_loss_times else None,
            "first_loaded_data": loaded_times[0].isoformat() if loaded_times else None,
            "first_running": running_times[0].isoformat() if running_times else None,
        },
        "counts": {
            "total_loss_lines": len(total_loss_times),
            "total_loss_lines_used_after_warmup": steps,
            "warmup_steps_dropped": warmup_steps,
            "running_lines": len(running_times),
            "loaded_data_lines": len(loaded_times),
        },
        "efficiency_from_total_loss_timestamps": {
            "eff_window_start": eff_start.isoformat() if eff_start else None,
            "eff_window_end": eff_end.isoformat() if eff_end else None,
            "eff_window_seconds": eff_seconds if eff_start and eff_end else None,
            "sec_per_logged_step": sec_per_step,
            "logged_steps_per_sec": steps_per_sec,
            "interval_sec_mean": interval_mean,
            "interval_sec_min": interval_min,
            "interval_sec_max": interval_max,
        },
        "overall_runtime_seconds": overall_seconds,
    }

def pretty_print(name: str, res: dict):
    eff = res["efficiency_from_total_loss_timestamps"]
    cnt = res["counts"]

    print(f"\n=== {name} ===")
    print(f"log: {res['log_path']}")
    print(f"total_loss_lines: {cnt['total_loss_lines']}  used(after warmup): {cnt['total_loss_lines_used_after_warmup']}  warmup_dropped: {cnt['warmup_steps_dropped']}")
    print(f"overall_runtime_seconds: {res['overall_runtime_seconds']}")
    print("efficiency (based on Total loss timestamps, post-warmup):")
    for k in ["eff_window_seconds", "sec_per_logged_step", "logged_steps_per_sec", "interval_sec_mean", "interval_sec_min", "interval_sec_max"]:
        print(f"  - {k}: {eff.get(k)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_a", required=True, help="path to adapter A log file")
    ap.add_argument("--log_b", required=True, help="path to adapter B log file")
    ap.add_argument("--warmup_steps", type=int, default=0, help="drop first N Total-loss records before timing")
    ap.add_argument(
        "--start_anchor",
        choices=["auto", "running", "loaded", "first_total"],
        default="auto",
        help="how to define overall start time window",
    )
    ap.add_argument("--out_json", default=None, help="optional path to save json report")
    args = ap.parse_args()

    res_a = analyze_log(Path(args.log_a), warmup_steps=args.warmup_steps, start_anchor=args.start_anchor)
    res_b = analyze_log(Path(args.log_b), warmup_steps=args.warmup_steps, start_anchor=args.start_anchor)

    pretty_print("A", res_a)
    pretty_print("B", res_b)

    # A vs B delta (main metrics)
    def get(x, key):
        return x["efficiency_from_total_loss_timestamps"].get(key)

    delta = {}
    for k in ["sec_per_logged_step", "logged_steps_per_sec", "eff_window_seconds"]:
        a = get(res_a, k)
        b = get(res_b, k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a is not None and b is not None:
            delta[k] = a - b
        else:
            delta[k] = None

    print("\n=== DELTA (A - B) ===")
    for k, v in delta.items():
        print(f"{k}: {v}")

    out = {"A": res_a, "B": res_b, "delta_A_minus_B": delta}
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n[DONE] wrote json:", args.out_json)

if __name__ == "__main__":
    main()

