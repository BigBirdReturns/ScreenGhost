"""v0.1 release-cut guards: artifacts exist, claim boundaries held."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


def test_release_artifacts_exist():
    for rel in ("CHANGELOG.md", "RELEASE_NOTES_v0.1.md", "docs/CLAIM_BOUNDARIES.md",
                "examples/verify_release.py", "examples/receipts/release_v0.1.txt"):
        assert os.path.exists(os.path.join(ROOT, rel)), rel


def test_all_issue_templates_present():
    d = os.path.join(ROOT, ".github", "ISSUE_TEMPLATE")
    for t in ("adapter_failure.md", "resolver_failure.md", "product_ui_issue.md",
              "receipt_reproducibility.md", "claim_boundary.md"):
        assert os.path.exists(os.path.join(d, t)), t


def test_claim_boundaries_lists_every_forbidden_claim():
    txt = _read("docs/CLAIM_BOUNDARIES.md").lower()
    for phrase in ("no hardware proof", "no live-seller proof",
                   "no business", "universal real-app", "merchant thai"):
        assert phrase in txt


def test_release_receipt_passes_and_disclaims():
    txt = _read("examples/receipts/release_v0.1.txt")
    assert "release_verdict" in txt and "PASS" in txt
    low = txt.lower()
    assert "not hardware proof" in low and "not business proof" in low


def test_release_notes_and_changelog_disclaim_claims():
    for rel in ("RELEASE_NOTES_v0.1.md", "CHANGELOG.md"):
        low = _read(rel).lower()
        assert "not hardware proof" in low or "no hardware proof" in low
        assert "business" in low and "frozen" in low


def test_changelog_records_the_determinism_fix():
    low = _read("CHANGELOG.md").lower()
    assert "salted" in low and "hash()" in low   # the RC0 bug is on the record
