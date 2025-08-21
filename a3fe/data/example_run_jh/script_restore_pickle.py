"""
Restore Pickle Files from Backups

This script restores pickle files from their .backup versions.
"""

import os
import glob
import shutil
import argparse
from pathlib import Path
from typing import List


def find_backup_files(base_dir: str) -> List[str]:
    """Find all .backup files in the directory tree."""
    patterns = [
        "**/*.pkl.backup",
        "**/Calculation.pkl.backup",
        "**/Leg.pkl.backup", 
        "**/Stage.pkl.backup",
        "**/LamWindow.pkl.backup",
        "**/Simulation.pkl.backup"
    ]
    
    backup_files = []
    for pattern in patterns:
        found = glob.glob(os.path.join(base_dir, pattern), recursive=True)
        backup_files.extend(found)
    
    # Remove duplicates and sort
    unique_files = list(set(backup_files))
    unique_files.sort()
    
    return unique_files


def restore_single_backup(backup_path: str, dry_run: bool = False) -> bool:
    """
    Restore a single backup file.
    Returns True if restoration was successful.
    """
    # Get the original file path by removing .backup extension
    original_path = backup_path.replace('.backup', '')
    
    print(f"  Backup: {backup_path}")
    print(f"  Target: {original_path}")
    
    # Check if backup exists
    if not os.path.isfile(backup_path):
        print(f"  ❌ Backup file does not exist")
        return False
    
    # Check if we would overwrite an existing file
    if os.path.isfile(original_path):
        print(f"  ⚠️  Will overwrite existing file")
    
    if not dry_run:
        try:
            # Copy backup to original location
            shutil.copy2(backup_path, original_path)
            print(f"  ✅ Successfully restored")
            return True
        except Exception as e:
            print(f"  ❌ Error restoring: {e}")
            return False
    else:
        print(f"  [DRY RUN] Would restore this file")
        return True


def restore_all_backups(base_dir: str, dry_run: bool = False) -> dict:
    """
    Restore all backup files in a directory.
    Returns statistics about the restoration.
    """
    print(f"Searching for backup files in: {base_dir}")
    
    backup_files = find_backup_files(base_dir)
    
    if not backup_files:
        print("No backup files found.")
        return {"total": 0, "restored": 0, "failed": 0}
    
    print(f"Found {len(backup_files)} backup files")
    if dry_run:
        print("=== DRY RUN MODE - NO FILES WILL BE MODIFIED ===")
    
    print("=" * 60)
    
    stats = {"total": len(backup_files), "restored": 0, "failed": 0}
    
    for backup_file in backup_files:
        print(f"\nProcessing: {os.path.basename(backup_file)}")
        
        if restore_single_backup(backup_file, dry_run):
            stats["restored"] += 1
        else:
            stats["failed"] += 1
    
    return stats


def clean_backup_files(base_dir: str, dry_run: bool = False) -> int:
    """
    Remove all backup files after successful restoration.
    Returns number of backup files removed.
    """
    backup_files = find_backup_files(base_dir)
    
    if not backup_files:
        print("No backup files to clean.")
        return 0
    
    print(f"\nCleaning {len(backup_files)} backup files...")
    if dry_run:
        print("=== DRY RUN MODE - NO FILES WILL BE DELETED ===")
    
    removed_count = 0
    for backup_file in backup_files:
        print(f"  Removing: {backup_file}")
        if not dry_run:
            try:
                os.remove(backup_file)
                removed_count += 1
                print(f"    ✅ Deleted")
            except Exception as e:
                print(f"    ❌ Error deleting: {e}")
        else:
            print(f"    [DRY RUN] Would delete this file")
            removed_count += 1
    
    return removed_count


def list_backup_files(base_dir: str):
    """List all backup files without restoring them."""
    backup_files = find_backup_files(base_dir)
    
    if not backup_files:
        print("No backup files found.")
        return
    
    print(f"Found {len(backup_files)} backup files:")
    print("=" * 60)
    
    for backup_file in backup_files:
        original_file = backup_file.replace('.backup', '')
        
        # Get file info
        backup_stat = os.stat(backup_file)
        backup_size = backup_stat.st_size
        backup_time = backup_stat.st_mtime
        
        print(f"Backup: {backup_file}")
        print(f"  -> {original_file}")
        print(f"  Size: {backup_size:,} bytes")
        print(f"  Modified: {backup_time}")
        
        if os.path.isfile(original_file):
            orig_stat = os.stat(original_file)
            print(f"  Original exists: {orig_stat.st_size:,} bytes")
        else:
            print(f"  Original: MISSING")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Restore A3FE pickle files from backup copies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python restore_pickle_backups.py --list /path/to/calculation
  python restore_pickle_backups.py --dry-run /path/to/calculation
  python restore_pickle_backups.py /path/to/calculation
  python restore_pickle_backups.py --clean /path/to/calculation
        """
    )
    
    parser.add_argument(
        "directory",
        help="Directory containing backup files to restore"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be restored without actually doing it"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all backup files without restoring them"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all backup files after successful restoration"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        return 1
    
    if args.list:
        # Just list backup files
        list_backup_files(args.directory)
    else:
        # Restore backup files
        stats = restore_all_backups(args.directory, args.dry_run)
        
        # Print summary
        print("=" * 60)
        print("RESTORATION SUMMARY:")
        print(f"Total backup files found: {stats['total']}")
        print(f"Successfully restored: {stats['restored']}")
        print(f"Failed to restore: {stats['failed']}")
        
        if not args.dry_run and stats['restored'] > 0:
            print(f"\n✅ {stats['restored']} pickle files have been restored from backups!")
            
            if args.clean:
                # Clean up backup files
                removed = clean_backup_files(args.directory, False)
                print(f"🧹 Cleaned up {removed} backup files")
        elif args.dry_run:
            print("\nTo actually restore files, run without --dry-run")
    
    return 0


if __name__ == "__main__":
    exit(main())