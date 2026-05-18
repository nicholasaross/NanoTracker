#!/usr/bin/env python3
"""Pull a NanoTracker capture batch from the Nano to the local box.

By default pulls the most recent `session_*` directory under
`/home/claude/NanoTracker/output/` on the Nano into `./output/` on the
local box, preserving the session-dir structure so paths in the session
HTML / JSON keep resolving locally.

Defaults match the standard CLAUDE.md SSH setup (user=claude, host=nano,
key=~/.ssh/nanotracker_claude).  Designed to run on the Windows dev box
where the built-in OpenSSH client provides `ssh` and `scp`; works equally
on Linux / macOS.

Usage:
    python scripts/pull_session.py                       # latest session -> ./output
    python scripts/pull_session.py --dry-run             # show plan, don't pull
    python scripts/pull_session.py --session session_20260518_141443
    python scripts/pull_session.py --target D:\\Captures
    python scripts/pull_session.py --only-main           # skip thumbs / HTML

Exits non-zero on SSH / scp failure.  Idempotent: re-running over an
existing local copy overwrites with the latest remote state (scp merges).
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_HOST = "nano"
DEFAULT_USER = "claude"
DEFAULT_KEY = "~/.ssh/nanotracker_claude"
DEFAULT_REMOTE_PARENT = "/home/claude/NanoTracker/output"
DEFAULT_LOCAL_PARENT = "./output"


def ssh_run(host, user, key, cmd, check=True):
    """Run `cmd` on the Nano via SSH and return stripped stdout."""
    args = ["ssh", "-i", key, "-o", "BatchMode=yes",
            "{}@{}".format(user, host), cmd]
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    if check and proc.returncode != 0:
        sys.stderr.write("[pull] ssh failed (exit {}):\n{}\n".format(
            proc.returncode, proc.stderr.strip()))
        sys.exit(proc.returncode or 1)
    return proc.stdout.strip()


def find_latest_session(host, user, key, remote_parent):
    cmd = "ls -1 {} 2>/dev/null | grep '^session_' | sort | tail -1".format(
        shlex.quote(remote_parent))
    s = ssh_run(host, user, key, cmd)
    if not s:
        sys.exit("[pull] No session_* directories found in {}:{}".format(
            host, remote_parent))
    return s


def remote_inventory(host, user, key, remote_path):
    """Return (total_bytes, file_count, jpeg_count_by_kind) for the session."""
    # POSIX-portable inventory: total size in bytes, total file count, and
    # counts split by filename suffix.  -L follows symlinks (none expected
    # but safe).  Single ssh round-trip keeps this snappy.
    cmd = (
        "cd {p} 2>/dev/null && "
        "du -sb . 2>/dev/null | awk '{{print \"BYTES \" $1}}' ; "
        "find . -maxdepth 1 -type f | wc -l | awk '{{print \"FILES \" $1}}' ; "
        "find . -maxdepth 1 -name '*_main_*.jpg' | wc -l | awk '{{print \"MAIN \" $1}}' ; "
        "find . -maxdepth 1 -name '*_hq.jpg'     | wc -l | awk '{{print \"HQ \" $1}}' ; "
        "find . -maxdepth 1 -name '*.jsonl'      | wc -l | awk '{{print \"JSONL \" $1}}'"
    ).format(p=shlex.quote(remote_path))
    out = ssh_run(host, user, key, cmd, check=False)
    inv = {"BYTES": 0, "FILES": 0, "MAIN": 0, "HQ": 0, "JSONL": 0}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in inv:
            try:
                inv[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return inv


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "{:.1f} {}".format(n, unit)
        n /= 1024.0


def scp_pull(host, user, key, remote_path, local_parent, only_main, dry_run):
    """scp the remote session dir into local_parent.

    With --only-main, we issue a separate scp for just the main snaps and
    JSON metadata, avoiding the (small but real) cost of pulling thumbs +
    HTML when the user only wants raw frames for ALPR / post-processing.
    """
    local_parent.mkdir(parents=True, exist_ok=True)
    remote_target = "{}@{}:{}".format(user, host, remote_path)

    if only_main:
        # Pull JSONs + main snaps explicitly; skip thumbs (<id>.jpg) and HQ
        # (<id>_hq.jpg).  Per-pattern scp keeps the include list simple
        # without needing rsync (not on Windows OpenSSH by default).
        session_name = remote_path.rstrip("/").rsplit("/", 1)[-1]
        local_session = local_parent / session_name
        local_session.mkdir(parents=True, exist_ok=True)
        patterns = ["*_main_*.jpg", "*.json", "*.jsonl", "*_summary.html",
                    "index.html"]
        commands = []
        for pat in patterns:
            src = "{}@{}:{}/{}".format(user, host, remote_path, pat)
            commands.append(["scp", "-i", key, "-p", "-q", src, str(local_session)])
    else:
        commands = [["scp", "-i", key, "-r", "-p",
                     remote_target, str(local_parent)]]

    for args in commands:
        if dry_run:
            print("[dry-run]", " ".join(shlex.quote(a) for a in args))
            continue
        proc = subprocess.run(args)
        # scp returns nonzero if no files match a wildcard; for --only-main
        # we accept that (a session with no main snaps is still valid).
        if proc.returncode != 0 and not only_main:
            sys.exit("[pull] scp failed (exit {})".format(proc.returncode))


def main():
    p = argparse.ArgumentParser(
        description=__doc__.strip().split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    p.add_argument("--host", default=DEFAULT_HOST,
                   help="SSH host or alias (default: {})".format(DEFAULT_HOST))
    p.add_argument("--user", default=DEFAULT_USER,
                   help="SSH user (default: {})".format(DEFAULT_USER))
    p.add_argument("--key", default=DEFAULT_KEY,
                   help="SSH private key (default: {})".format(DEFAULT_KEY))
    p.add_argument("--remote-parent", default=DEFAULT_REMOTE_PARENT,
                   help="Parent output directory on the Nano")
    p.add_argument("--target", default=DEFAULT_LOCAL_PARENT,
                   help="Local parent directory to receive the session "
                        "(session subdir is created inside)")
    p.add_argument("--session", default=None,
                   help="Specific session label (default: latest)")
    p.add_argument("--only-main", action="store_true",
                   help="Pull only main-stream snaps + JSON metadata "
                        "(skip thumbnail / HQ crop / summary HTML)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the scp invocation and the remote inventory, "
                        "do not transfer")
    args = p.parse_args()

    key = os.path.expanduser(args.key)
    if not Path(key).exists():
        sys.exit("[pull] SSH key not found: {}".format(key))

    session = args.session or find_latest_session(
        args.host, args.user, key, args.remote_parent)
    remote_path = "{}/{}".format(args.remote_parent.rstrip("/"), session)
    local_parent = Path(args.target).expanduser().resolve()

    inv = remote_inventory(args.host, args.user, key, remote_path)
    print("[pull] session: {}".format(session))
    print("[pull] remote:  {}@{}:{}".format(args.user, args.host, remote_path))
    print("[pull] target:  {}".format(local_parent))
    print("[pull] size:    {} across {} files ({} main snaps, {} HQ crops, {} jsonl)".format(
        human_bytes(inv["BYTES"]), inv["FILES"], inv["MAIN"], inv["HQ"], inv["JSONL"]))
    if args.only_main:
        print("[pull] mode:    --only-main (skipping thumbs + HQ + HTML)")

    scp_pull(args.host, args.user, key, remote_path, local_parent,
             args.only_main, args.dry_run)

    if not args.dry_run:
        landed = local_parent / session
        print("[pull] done -> {}".format(landed))
        html = landed / "{}_summary.html".format(session)
        if html.exists():
            print("[pull] open:   file:///{}".format(
                str(html).replace("\\", "/")))


if __name__ == "__main__":
    main()
