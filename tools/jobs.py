#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Look at a server's job store without the server.

    uv run tools/jobs.py list                      # newest first
    uv run tools/jobs.py show <id>
    uv run tools/jobs.py stats
    uv run tools/jobs.py reap --ttl-hours 24       # drop terminal records
    uv run tools/jobs.py sql "select ..."          # read-only escape hatch

The job store is sqlite, so an operator needs a way to read it the way `ls`
reads the content store - that readability is why the caches use percent-encoded
readable names rather than hashes, and a database should not quietly give it up.

Dependency-free on purpose (sqlite3 is stdlib) so it runs against a store on a
machine that has no nnseg checkout - a mounted Modal volume, a copied workdir, a
laptop being debugged over ssh. It duplicates no logic: it reads the same rows
`nnseg.jobstore` writes, and `reap` applies the same TTL the server does.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT = Path("/tmp/nnseg-serve/jobs.db")
TERMINAL = ("done", "failed", "cancelled")


def _open(path: Path, *, write: bool = False) -> sqlite3.Connection:
    if not path.exists():
        sys.exit(f"no job store at {path} (pass --db, or the server has not run yet)")
    # read-only unless a subcommand actually mutates, so `list` on a live
    # server's store can never disturb it
    uri = f"file:{path}?mode={'rw' if write else 'ro'}"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    return db


def _age(ts) -> str:
    if not ts:
        return "-"
    d = time.time() - float(ts)
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if d >= n:
            return f"{d / n:.0f}{unit}"
    return f"{d:.0f}s"


def cmd_list(a) -> None:
    rows = _open(a.db).execute(
        "SELECT id, task, state, cache_key, created, finished FROM jobs"
        + (" WHERE state = ?" if a.state else "")
        + " ORDER BY created DESC LIMIT ?",
        ((a.state, a.limit) if a.state else (a.limit,))).fetchall()
    if not rows:
        print("(no jobs)")
        return
    print(f"{'id':14}{'state':11}{'age':>6}  {'task':32}key")
    for r in rows:
        key = (r["cache_key"] or "")[:12]
        print(f"{r['id']:14}{r['state']:11}{_age(r['created']):>6}  "
              f"{(r['task'] or '')[:31]:32}{key}")


def cmd_show(a) -> None:
    row = _open(a.db).execute(
        "SELECT payload FROM jobs WHERE id = ?", (a.id,)).fetchone()
    if row is None:
        sys.exit(f"no job {a.id!r}")
    rec = json.loads(row["payload"])
    rec.pop("source_tokens", None)          # never stored, never printed
    print(json.dumps(rec, indent=2, sort_keys=True))


def cmd_stats(a) -> None:
    db = _open(a.db)
    counts = db.execute("SELECT state, COUNT(*) n FROM jobs GROUP BY state").fetchall()
    total = sum(r["n"] for r in counts)
    print(f"{total} records at {a.db}")
    for r in sorted(counts, key=lambda r: -r["n"]):
        print(f"  {r['state']:11}{r['n']:>6}")
    oldest = db.execute("SELECT MIN(created) c FROM jobs").fetchone()["c"]
    if oldest:
        print(f"  oldest     {_age(oldest):>6}")


def cmd_reap(a) -> None:
    db = _open(a.db, write=True)
    cutoff = time.time() - a.ttl_hours * 3600
    q = ",".join("?" * len(TERMINAL))
    rows = db.execute(
        f"SELECT id FROM jobs WHERE state IN ({q}) AND COALESCE(finished, created) < ?",
        (*TERMINAL, cutoff)).fetchall()
    ids = [r["id"] for r in rows]
    if a.dry_run:
        print(f"would drop {len(ids)} record(s)")
        for i in ids[:20]:
            print(f"  {i}")
        return
    if ids:
        db.execute(f"DELETE FROM jobs WHERE id IN ({','.join('?' * len(ids))})", ids)
        db.commit()
    print(f"dropped {len(ids)} record(s); their job directories are the "
          f"server's to remove")


def cmd_sql(a) -> None:
    if not a.query.lstrip().lower().startswith("select"):
        sys.exit("only SELECT is allowed here; use `reap` to delete")
    for row in _open(a.db).execute(a.query).fetchall():
        print("\t".join("" if v is None else str(v) for v in tuple(row)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", type=Path, default=DEFAULT, help=f"default {DEFAULT}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list", help="jobs, newest first")
    ls.add_argument("--state"); ls.add_argument("--limit", type=int, default=40)
    ls.set_defaults(fn=cmd_list)

    sh = sub.add_parser("show", help="one job's full record")
    sh.add_argument("id"); sh.set_defaults(fn=cmd_show)

    sub.add_parser("stats", help="counts by state").set_defaults(fn=cmd_stats)

    rp = sub.add_parser("reap", help="drop terminal records past a TTL")
    rp.add_argument("--ttl-hours", type=float, default=24.0)
    rp.add_argument("--dry-run", action="store_true")
    rp.set_defaults(fn=cmd_reap)

    sq = sub.add_parser("sql", help="read-only query")
    sq.add_argument("query"); sq.set_defaults(fn=cmd_sql)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
