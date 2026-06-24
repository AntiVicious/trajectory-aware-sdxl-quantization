#!/usr/bin/env python3
"""
Archive everything in gem/ that was last modified BEFORE April 1, 2026.

Protected items that stay regardless of mtime:
  - val2017/       (COCO reference, mtime=2017 but needed for FID)
  - val2017.zip    (also protected — let user delete manually if unwanted)
  - Any directory whose contents include ANY file modified on/after 2026-04-01
    (this catches long-lived folders like trajectory/ that contain recent work)

Default: dry-run.
Pass --execute to actually move.

Usage:
    python cleanup_by_date.py
    python cleanup_by_date.py --execute
"""
import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime

GEM = Path("/home/project/Documents/Maharshi/gem")
OLD_GEM = Path("/home/project/Documents/Maharshi/old_gem")

# Cutoff: move anything with mtime BEFORE this
CUTOFF = datetime(2026, 4, 1).timestamp()

# Always-keep items, regardless of mtime
PROTECTED = {
    "val2017",       # COCO FID reference — mtime is 2017 but permanent
    "val2017.zip",   # source zip — user can archive separately if desired
}


def dir_has_recent_content(path: Path, cutoff: float) -> bool:
    """Returns True if any file in the directory tree has mtime >= cutoff."""
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    if p.stat().st_mtime >= cutoff:
                        return True
                except OSError:
                    pass
    except OSError:
        pass
    return False


def dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def format_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    if not GEM.exists():
        print(f"ERROR: {GEM} does not exist.")
        sys.exit(1)

    cutoff_dt = datetime.fromtimestamp(CUTOFF)
    print(f"{'EXECUTE' if args.execute else 'DRY RUN'} mode.")
    print(f"Cutoff: anything with mtime BEFORE {cutoff_dt:%Y-%m-%d} moves to old_gem/")
    print(f"Protected: {sorted(PROTECTED)}")
    print()

    if args.execute:
        OLD_GEM.mkdir(parents=True, exist_ok=True)

    # Scan top level
    entries = sorted(GEM.iterdir(),
                     key=lambda p: (not p.is_dir(), p.name.lower()))

    to_keep = []
    to_move = []

    for p in entries:
        if p.name in PROTECTED:
            to_keep.append((p, "protected"))
            continue

        if p.is_dir():
            # Check if any file inside has been modified recently
            if dir_has_recent_content(p, CUTOFF):
                to_keep.append((p, "contains recent content"))
            else:
                to_keep_reason = None
                # Otherwise check top-level mtime as fallback
                try:
                    if p.stat().st_mtime >= CUTOFF:
                        to_keep.append((p, f"mtime {format_mtime(p.stat().st_mtime)}"))
                        continue
                except OSError:
                    pass
                to_move.append((p, "old, no recent content"))
        else:
            # File: just check mtime
            try:
                mt = p.stat().st_mtime
                if mt >= CUTOFF:
                    to_keep.append((p, f"mtime {format_mtime(mt)}"))
                else:
                    to_move.append((p, f"mtime {format_mtime(mt)}"))
            except OSError as e:
                print(f"  SKIP (stat failed): {p.name}: {e}")

    # Report
    total_keep_bytes = 0
    total_move_bytes = 0

    print("=" * 86)
    print(f"KEEP in gem/  ({len(to_keep)} items)")
    print("=" * 86)
    for p, reason in to_keep:
        sz = dir_size(p) if p.is_dir() else p.stat().st_size
        total_keep_bytes += sz
        tag = "[DIR]" if p.is_dir() else "     "
        print(f"  {tag} {human(sz):>10}  {p.name:<42}  ({reason})")

    print()
    print("=" * 86)
    print(f"MOVE to old_gem/  ({len(to_move)} items)")
    print("=" * 86)

    # Group for readability
    dirs_to_move = [(p, r) for p, r in to_move if p.is_dir()]
    ckpts_to_move = [(p, r) for p, r in to_move if p.is_file() and p.suffix == ".pt"]
    pys_to_move = [(p, r) for p, r in to_move if p.is_file() and p.suffix == ".py"]
    imgs_to_move = [(p, r) for p, r in to_move if p.is_file()
                    and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    others = [(p, r) for p, r in to_move
              if (p, r) not in dirs_to_move + ckpts_to_move + pys_to_move + imgs_to_move]

    def bytes_of(items):
        b = 0
        for p, _ in items:
            sz = dir_size(p) if p.is_dir() else p.stat().st_size
            b += sz
        return b

    if dirs_to_move:
        print(f"\n  Directories ({len(dirs_to_move)}, {human(bytes_of(dirs_to_move))}):")
        for p, r in dirs_to_move:
            sz = dir_size(p)
            total_move_bytes += sz
            print(f"    {human(sz):>10}  {p.name}/  ({r})")

    if ckpts_to_move:
        sub_total = bytes_of(ckpts_to_move)
        total_move_bytes += sub_total
        print(f"\n  .pt checkpoints ({len(ckpts_to_move)}, {human(sub_total)}):")
        for p, r in ckpts_to_move[:8]:
            print(f"    {human(p.stat().st_size):>10}  {p.name:<44}  ({r})")
        if len(ckpts_to_move) > 8:
            rest_size = sum(p.stat().st_size for p, _ in ckpts_to_move[8:])
            print(f"    ... and {len(ckpts_to_move) - 8} more  ({human(rest_size)})")

    if pys_to_move:
        sub_total = bytes_of(pys_to_move)
        total_move_bytes += sub_total
        print(f"\n  .py scripts ({len(pys_to_move)}, {human(sub_total)}):")
        for p, r in pys_to_move[:10]:
            print(f"    {human(p.stat().st_size):>10}  {p.name:<44}  ({r})")
        if len(pys_to_move) > 10:
            rest_size = sum(p.stat().st_size for p, _ in pys_to_move[10:])
            print(f"    ... and {len(pys_to_move) - 10} more  ({human(rest_size)})")

    if imgs_to_move:
        sub_total = bytes_of(imgs_to_move)
        total_move_bytes += sub_total
        print(f"\n  Images ({len(imgs_to_move)}, {human(sub_total)}):")
        for p, r in imgs_to_move[:5]:
            print(f"    {human(p.stat().st_size):>10}  {p.name:<44}  ({r})")
        if len(imgs_to_move) > 5:
            rest_size = sum(p.stat().st_size for p, _ in imgs_to_move[5:])
            print(f"    ... and {len(imgs_to_move) - 5} more  ({human(rest_size)})")

    if others:
        sub_total = bytes_of(others)
        total_move_bytes += sub_total
        print(f"\n  Other files ({len(others)}, {human(sub_total)}):")
        for p, r in others[:8]:
            sz = p.stat().st_size if p.is_file() else dir_size(p)
            print(f"    {human(sz):>10}  {p.name:<44}  ({r})")
        if len(others) > 8:
            print(f"    ... and {len(others) - 8} more")

    print()
    print("=" * 86)
    print("SUMMARY")
    print("=" * 86)
    print(f"  Total before:  {human(total_keep_bytes + total_move_bytes)}")
    print(f"  gem/ after:    {human(total_keep_bytes)}")
    print(f"  old_gem/:      {human(total_move_bytes)}")

    if not args.execute:
        print()
        print("DRY RUN complete. To execute:")
        print(f"  python {sys.argv[0]} --execute")
        return

    print()
    confirm = input(f"Move {len(to_move)} items ({human(total_move_bytes)}) to old_gem/? [yes/no]: ")
    if confirm.strip().lower() not in ("yes", "y"):
        print("Aborted.")
        return

    moved = 0
    errors = []
    for p, _ in to_move:
        dest = OLD_GEM / p.name
        try:
            if dest.exists():
                i = 2
                while (OLD_GEM / f"{p.name}.old{i}").exists():
                    i += 1
                dest = OLD_GEM / f"{p.name}.old{i}"
            shutil.move(str(p), str(dest))
            moved += 1
            if moved % 20 == 0:
                print(f"  moved {moved}/{len(to_move)}")
        except Exception as e:
            errors.append((p, str(e)))
            print(f"  FAIL: {p.name}: {e}")

    print(f"\nDone. Moved {moved}/{len(to_move)}.")
    if errors:
        print(f"\n{len(errors)} errors:")
        for p, err in errors:
            print(f"  {p.name}: {err}")


if __name__ == "__main__":
    main()
