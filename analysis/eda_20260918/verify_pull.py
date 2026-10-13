#!/usr/bin/env python3
# =============================================================================
# . RESPONSIBILITY: verify one pulled PID-baseline tree against its S3 size record, rebuild its
#   paired/env0 links by the sole-env rule, list them, and link it into /home/ubuntu/flightdock_corpora/<name>.
#   Outcome-blind: reads file counts, sizes, leg_meta and directory structure only.
# CLI: --name PID_s035 --tree <legs_full dir> --s3-objects N --s3-bytes N --ckpt <sha>
# SIDE-EFFECTS: creates <leg>/paired/env0 symlinks, <tree>/SYMLINKS_<name>.txt and the corpora link. Deletes nothing.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import collections
import glob
import json
import os
import sys

CORP = (_os.environ.get("FLIGHTDOCK_CORPORA", _REPO + "/corpora"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("--name", "--tree", "--ckpt"):
        ap.add_argument(k, required=True)
    ap.add_argument("--s3-objects", type=int, required=True)
    ap.add_argument("--s3-bytes", type=int, required=True)
    a = ap.parse_args(argv)
    files = [os.path.join(d, f) for d, _, fs in os.walk(a.tree) for f in fs]
    files = [f for f in files if not os.path.basename(f).startswith("SYMLINKS_")]
    nb = sum(os.path.getsize(f) for f in files)
    print("files %d (S3 %d) bytes %d (S3 %d) match=%s" % (len(files), a.s3_objects, nb, a.s3_bytes,
                                                           len(files) == a.s3_objects and nb == a.s3_bytes))
    legs = [d for d in sorted(glob.glob(os.path.join(a.tree, "B*", "*_*_*")))
            if os.path.isdir(d) and not os.path.basename(d).startswith("episode_") and os.path.basename(d).split("_")[1].isdigit()]
    metas = [json.load(open(os.path.join(l, "leg_meta.json"))) for l in legs if os.path.exists(os.path.join(l, "leg_meta.json"))]
    ck = collections.Counter(m.get("checkpoint_sha256") for m in metas)
    seeds = sorted({m["seed"] for m in metas})
    print("legs %d cells %d leg_meta %d results %d seeds %d-%d (%d) policy checkpoint %s expected %s" % (
        len(legs), len({os.path.dirname(l) for l in legs}), len(metas),
        sum(len(glob.glob(os.path.join(l, "results_eid*.json"))) == 1 for l in legs), seeds[0], seeds[-1], len(seeds),
        {k[:8] if k else k: v for k, v in ck.items()}, a.ckpt[:8]))
    rows, made = [], 0
    for leg in legs:
        e = glob.glob(os.path.join(leg, "env[0-9]*"))
        if len(e) != 1:
            print("REFUSE: %s has %d env dirs" % (leg, len(e)))
            return 3
        tgt, lk = "../" + os.path.basename(e[0]), os.path.join(leg, "paired", "env0")
        os.makedirs(os.path.dirname(lk), exist_ok=True)
        if os.path.islink(lk):
            if os.readlink(lk) != tgt:
                print("REFUSE: %s points to %s" % (lk, os.readlink(lk)))
                return 3
        else:
            os.symlink(tgt, lk)
            made += 1
        rows.append(os.path.relpath(lk, a.tree) + "\t" + tgt)
    open(os.path.join(a.tree, "SYMLINKS_%s.txt" % a.name), "w").write("\n".join(rows) + "\n")
    ok = sum(os.path.isdir(os.path.join(a.tree, r.split("\t")[0])) for r in rows)
    print("paired/env0 links made %d, resolving %d of %d" % (made, ok, len(rows)))
    dst = os.path.join(CORP, a.name)
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(a.tree), dst)
    print("corpora link %s -> %s %s" % (dst, os.readlink(dst), "ok" if os.path.isdir(dst) else "BROKEN"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
