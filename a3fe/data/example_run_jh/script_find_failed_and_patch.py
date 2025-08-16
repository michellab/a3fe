#!/usr/bin/env python3
from __future__ import annotations
import re
import sys
from pathlib import Path

# ---- EDIT THESE ROOTS for your environment ----
# PROJECT_ROOT = Path("/home/jjhuang/project/jjhuang/fep_workflows")
SCRATCH_ROOT = Path("/home/jjhuang/scratch/jjhuang/fep_workflows")
# ------------------------------------------------

ERR_SIGNATURE = "The ligand has a non-zero charge"
SYS_DIR_RE = re.compile(r"system_(\d+)$")

def find_failed_system_nums(scratch_root: Path) -> list[int]:
    """Scan SCRATCH_ROOT/system_{n}/*.err for the PME/charge error; return sorted unique nums."""
    failed = set()
    if not scratch_root.exists():
        return []
    for sysdir in scratch_root.iterdir():
        m = SYS_DIR_RE.match(sysdir.name)
        if not (sysdir.is_dir() and m):
            continue
        for errf in sysdir.glob("*.err"):
            try:
                txt = errf.read_text(errors="ignore")
            except Exception:
                continue
            if ERR_SIGNATURE in txt:
                failed.add(int(m.group(1)))
                break
    return sorted(failed)

def patch_cfg(cfg_path: Path, dry_run: bool=False) -> str:
    """
    Ensure 'cutoff type = PME' in template_config.cfg.
    - If a 'cutoff type =' line exists, replace it with 'cutoff type = PME' (case-insensitive).
    - Otherwise, append it at the end.
    Returns a status string.
    """
    if not cfg_path.exists():
        return f"[MISS] {cfg_path} (no such file)"

    try:
        original = cfg_path.read_text()
    except Exception as e:
        return f"[ERROR] read {cfg_path}: {e}"

    lines = original.splitlines()
    changed = False
    found_line = False
    out_lines = []

    for line in lines:
        if re.match(r"^\s*cutoff\s*type\s*=", line, flags=re.IGNORECASE):
            found_line = True
            if line.strip() != "cutoff type = PME":
                out_lines.append("cutoff type = PME")
                changed = True
            else:
                out_lines.append(line)
            continue
        out_lines.append(line)

    if not found_line:
        out_lines.append("cutoff type = PME")
        changed = True

    if not changed:
        return f"[SKIP] Already PME: {cfg_path}"

    if dry_run:
        return f"[DRY] Would update: {cfg_path}"

    # Write with backup
    try:
        cfg_path.with_suffix(cfg_path.suffix + ".bak").write_text(original)
        cfg_path.write_text("\n".join(out_lines) + "\n")
        return f"[OK] Patched: {cfg_path}"
    except Exception as e:
        return f"[ERROR] write {cfg_path}: {e}"

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Find systems that failed due to PME/charge error and set 'cutoff type = PME' in their template_config.cfg.")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing files.")
    ap.add_argument("--list-only", action="store_true", help="Only list failed systems; do not patch.")
    args = ap.parse_args(argv)

    nums = find_failed_system_nums(SCRATCH_ROOT)
    if not nums:
        print("[INFO] No systems found with the PME/charge error.")
        return

    print(f"[INFO] Failed systems (PME/charge): {nums}")

    if args.list_only:
        return

    for n in nums:
        proj_cfg = SCRATCH_ROOT / f"system_{n}" / "input" / "template_config.cfg"
        print(patch_cfg(proj_cfg, dry_run=args.dry_run))

if __name__ == "__main__":
    main(sys.argv[1:])
