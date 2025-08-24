"""
Frontend patch verification for DNNE.

This module checks that modified npm packages in the frontend match our patched versions.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import filecmp


def verify_frontend_patches() -> bool:
    """
    Verify all frontend patches are correctly applied.
    
    Returns:
        True if all patches are verified or no patches exist, False if verification fails
    """
    # Get the frontend directory path
    backend_dir = Path(__file__).parent.parent  # DNNE-UI directory
    frontend_dir = backend_dir.parent / "DNNE-UI-Frontend"
    patches_file = frontend_dir / "dnne_patches" / "PATCHES.json"
    
    if not patches_file.exists():
        logging.info("No frontend patches to verify (PATCHES.json not found)")
        return True
    
    # Import the verification script from frontend
    sys.path.insert(0, str(frontend_dir / "dnne_patches"))
    try:
        from dnne_patches import verify_all_patches
    except ImportError as e:
        logging.error(f"Failed to import patch verification script: {e}")
        return False
    finally:
        sys.path.pop(0)
    
    # Run verification
    mismatches = verify_all_patches(str(frontend_dir))
    
    if not mismatches:
        logging.info("✓ All frontend patches verified successfully")
        return True
    
    # Log detailed error information
    logging.error("=" * 70)
    logging.error("FRONTEND PATCH VERIFICATION FAILED")
    logging.error("=" * 70)
    logging.error(f"{len(mismatches)} patch(es) do not match expected state:")
    logging.error("")
    
    for mismatch in mismatches:
        logging.error(f"Package: {mismatch['package']}")
        logging.error(f"  File: {mismatch['file']}")
        logging.error(f"  Issue: {mismatch['issue']}")
        logging.error("")
        
        # Provide detailed instructions based on the issue
        if "Timestamp mismatch" in mismatch['issue']:
            # Check if the current file matches either orig or patched
            target_file = Path(mismatch['target_path'])
            patch_file = Path(mismatch['patch_path'])
            orig_file = Path(mismatch['patch_path'].replace('/litegraph.es.js', '/orig_litegraph.es.js'))
            
            logging.error("  Paths:")
            logging.error(f"    Original (what we started with): {orig_file}")
            logging.error(f"    Patched (our modified version):   {patch_file}")
            logging.error(f"    Current (installed in node_modules): {target_file}")
            logging.error("")
            
            if orig_file.exists() and target_file.exists():
                if filecmp.cmp(orig_file, target_file, shallow=False):
                    logging.error("  ✓ The current file matches the original.")
                    logging.error("    Action: Simply copy our patched version over it:")
                    logging.error(f"      cp {patch_file} {target_file}")
                    logging.error("    Then update the timestamp in PATCHES.json")
                else:
                    logging.error("  ⚠ The current file differs from both original and patched versions.")
                    logging.error("    This likely means the package was updated.")
                    logging.error("    Action: You need to manually merge the changes:")
                    logging.error("      1. Review differences between original and current")
                    logging.error("      2. Apply our patches to the new version")
                    logging.error("      3. Update files in dnne_patches/")
                    logging.error("      4. Update timestamp in PATCHES.json")
            else:
                logging.error("  Action: Copy the patched file and update PATCHES.json:")
                logging.error(f"    cp {patch_file} {target_file}")
        logging.error("")
    
    logging.error("-" * 70)
    logging.error("To bypass this check and start anyway, use: --ignore-patch-errors")
    logging.error("=" * 70)
    
    return False