"""Filesystem anchors for tests that load scripts or recorded fixtures.

Tests are grouped in subdirectories that mirror ``src/looped_cdb``, so anchoring
on these constants keeps them independent of how deeply a test file is nested.
"""

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
EXPORTERS_DIR = SCRIPTS_DIR / "exporters"
SRC_DIR = REPO_ROOT / "src"
FIXTURES_DIR = TESTS_DIR / "fixtures"
