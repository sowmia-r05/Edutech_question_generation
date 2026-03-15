"""
apply_fixes.py
==============
Run this from your project root to apply two fixes:

Fix 1 — question_generator_v2.py
  Handles Gemini returning "N/A" / null / non-integer for page_number.
  Before: page_number=int(d.get("page_number", 0))  ← crashes on "N/A"
  After:  page_number=_safe_int(d.get("page_number", 0))

Fix 2 — Prints a clear checklist of which files you still need to replace.

Usage:
    python apply_fixes.py
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).parent
QG_PATH = ROOT / "src" / "generators" / "question_generator_v2.py"


def safe_int_helper() -> str:
    return '''
def _safe_int(value, default: int = 0) -> int:
    """Convert value to int safely — handles None, 'N/A', floats, strings."""
    if value is None:
        return default
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default

'''


def fix_question_generator():
    if not QG_PATH.exists():
        print(f"❌ Not found: {QG_PATH}")
        return False

    with open(QG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if already fixed
    if "_safe_int" in content:
        print("✅ question_generator_v2.py already patched")
        return True

    # 1. Add _safe_int helper after the imports / before the first function
    insert_after = "def _is_generic_sub_subject"
    if insert_after not in content:
        print(f"⚠️  Could not find insertion point in {QG_PATH}")
        return False

    content = content.replace(
        insert_after,
        safe_int_helper() + insert_after,
        1
    )

    # 2. Replace the crashing line
    old = 'page_number=int(d.get("page_number", 0)),'
    new = 'page_number=_safe_int(d.get("page_number", 0)),'

    if old not in content:
        # Try without trailing comma
        old2 = 'page_number=int(d.get("page_number", 0))'
        new2 = 'page_number=_safe_int(d.get("page_number", 0))'
        if old2 in content:
            content = content.replace(old2, new2)
            print("✅ Fixed page_number conversion (no trailing comma variant)")
        else:
            print("⚠️  page_number line not found — may already be fixed or different format")
    else:
        content = content.replace(old, new)
        print("✅ Fixed page_number int() crash → _safe_int()")

    with open(QG_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    return True


def check_files():
    print("\n" + "=" * 60)
    print("  FILE REPLACEMENT CHECKLIST")
    print("=" * 60)

    files_to_check = [
        ("src/generators/image_generator.py",
         "IMAGEN_MODELS",
         "Replace with the image_generator.py from Claude outputs"),

        ("src/generators/programmatic_image_generator.py",
         "ProgrammaticImageGenerator",
         "COPY this new file from Claude outputs to src/generators/"),

        ("src/core/qdrant_client_wrapper.py",
         "_upsert_points_with_retry",
         "Replace with qdrant_client_wrapper.py from Claude outputs"),

        ("generate_questions.py",
         "ProgrammaticImageGenerator",
         "Replace with generate_questions.py from Claude outputs"),
    ]

    all_good = True
    for rel_path, check_str, instruction in files_to_check:
        full_path = ROOT / rel_path
        if not full_path.exists():
            print(f"  ❌ MISSING : {rel_path}")
            print(f"     → {instruction}")
            all_good = False
        else:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            if check_str in content:
                print(f"  ✅ OK      : {rel_path}")
            else:
                print(f"  ⚠️  OUTDATED: {rel_path}")
                print(f"     → {instruction}")
                all_good = False

    print("=" * 60)
    if all_good:
        print("  All files up to date! ✅")
    else:
        print("  Replace the flagged files, then run again.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("\nApplying fixes...\n")
    fix_question_generator()
    check_files()