#!/usr/bin/env python3
"""
Validate CSI recording and labeling system (v2 trial-centric).
Usage: python validate.py [--host localhost] [--port 8000]
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def req(base: str, method: str, path: str, body=None):
    url = f"{base}{path}"
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(url, data=data, method=method)
    if data:
        r.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(r)
    return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}"

    passed = 0
    failed = 0
    skipped = 0
    trial_id = None

    def ok(msg):
        nonlocal passed
        passed += 1
        print(f"  \u2713  {msg}")

    def fail(msg):
        nonlocal failed
        failed += 1
        print(f"  \u2717  {msg}")

    def skip(msg):
        nonlocal skipped
        skipped += 1
        print(f"  ~  {msg}")

    print(f"\nValidating CSI recording system (v2) \u2014 {base}\n")

    # ── Reachability ─────────────────────────────────────────────────
    print("  Backend")
    try:
        req(base, "GET", "/api/stats")
        ok("Backend reachable")
    except Exception as e:
        fail(f"Backend unreachable: {e}")
        print(f"\n  0/{1} checks passed")
        sys.exit(1)

    # ── Recording ─────────────────────────────────────────────────────
    print("\n  Recording")
    try:
        r = req(base, "POST", "/api/recording/start", {
            "activity": "walking",
            "subject_id": "S01",
            "environment_id": "lab-alpha",
            "topology_id": "topo-10node",
            "tags": ["validation"],
            "notes": "validation test",
        })
        assert r.get("recording") is True
        trial_id = r["trial_id"]
        ok(f"Started trial {trial_id}")
    except Exception as e:
        fail(f"Start recording: {e}")

    try:
        r = req(base, "GET", "/api/recording/status")
        assert r.get("recording") is True
        ok("Recording status active")
    except Exception as e:
        fail(f"Recording status: {e}")

    time.sleep(1)

    # ── Annotations ───────────────────────────────────────────────────
    print("\n  Annotations")

    try:
        tax = req(base, "GET", "/api/annotations/taxonomy")
        n_cat = len(tax)
        n_act = sum(len(v) for v in tax.values())
        assert n_cat >= 5
        ok(f"Taxonomy: {n_cat} categories, {n_act} activities")
    except Exception as e:
        fail(f"Taxonomy: {e}")

    try:
        acts = req(base, "GET", "/api/annotations/activities")
        assert len(acts) >= 10
        ok(f"Activities list: {len(acts)} entries")
    except Exception as e:
        fail(f"Activities: {e}")

    point_id = None
    now = time.time()
    try:
        r = req(base, "POST", "/api/annotations", {
            "fine_label": "walking", "start_ts": now
        })
        point_id = r["annotation_id"]
        assert "trial_id" in r
        ok("POST point annotation with trial_id")
    except Exception as e:
        fail(f"Point annotation: {e}")

    interval_id = None
    try:
        r = req(base, "POST", "/api/annotations", {
            "fine_label": "sitting", "start_ts": now + 1, "end_ts": now + 4
        })
        interval_id = r["annotation_id"]
        ok("POST interval annotation")
    except Exception as e:
        fail(f"Interval annotation: {e}")

    try:
        anns = req(base, "GET", "/api/annotations")
        assert len(anns) >= 2
        ok(f"GET annotations: {len(anns)} records")
    except Exception as e:
        fail(f"GET annotations: {e}")

    if interval_id:
        try:
            req(base, "DELETE", f"/api/annotations/{interval_id}")
            ok("DELETE annotation")
        except Exception as e:
            fail(f"DELETE annotation: {e}")

    # ── Stop recording ────────────────────────────────────────────────
    try:
        r = req(base, "POST", "/api/recording/stop")
        assert r.get("recording") is False
        ok("Stopped recording")
    except Exception as e:
        fail(f"Stop recording: {e}")

    # ── Data files ────────────────────────────────────────────────────
    print("\n  Data files")

    if trial_id:
        # Find the trial dir by trial_id
        trials_dir = Path("data/trials")
        trial_dir = None
        if trials_dir.exists():
            for d in trials_dir.iterdir():
                if d.is_dir() and d.name.startswith(trial_id):
                    trial_dir = d
                    break

        if trial_dir and trial_dir.exists():
            ok(f"Trial dir: {trial_dir.name}")

            if (trial_dir / "trial.json").exists():
                meta = json.loads((trial_dir / "trial.json").read_text())
                assert meta["trial_id"] == trial_id
                assert "clock_sync" in meta
                ok("trial.json valid with clock_sync")
            else:
                fail("trial.json missing")

            raw = trial_dir / "raw" / "csi_raw.jsonl.zst"
            if raw.exists():
                ok(f"raw/csi_raw.jsonl.zst exists ({raw.stat().st_size} bytes)")
            else:
                skip("raw data: no CSI received during test")

            cal = trial_dir / "calibrated" / "csi_cal.jsonl.zst"
            if cal.exists():
                ok(f"calibrated/csi_cal.jsonl.zst exists ({cal.stat().st_size} bytes)")
            else:
                skip("calibrated data: no CSI received during test")

            ann = trial_dir / "annotations.jsonl"
            if ann.exists():
                ok("annotations.jsonl exists")
            else:
                skip("annotations.jsonl: none created in trial dir")

            # Check manifest
            manifest = Path("data/manifest.jsonl")
            if manifest.exists():
                found = False
                with open(manifest) as f:
                    for line in f:
                        if trial_id in line:
                            found = True
                            break
                if found:
                    ok("Trial in manifest.jsonl")
                else:
                    fail("Trial missing from manifest.jsonl")
            else:
                fail("manifest.jsonl missing")
        else:
            fail(f"Trial dir not found for {trial_id}")

    # ── Metadata endpoints ────────────────────────────────────────────
    print("\n  Metadata endpoints")
    for endpoint in ["/api/environments", "/api/subjects", "/api/topologies"]:
        try:
            r = req(base, "GET", endpoint)
            ok(f"GET {endpoint}: {len(r)} entries")
        except Exception as e:
            fail(f"GET {endpoint}: {e}")

    # ── Summary ───────────────────────────────────────────────────────
    total = passed + failed + skipped
    print(f"\n{'─' * 50}")
    suffix = f"  ({skipped} skipped)" if skipped else ""
    print(f"  {passed}/{total} checks passed, {failed} failed{suffix}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
