from __future__ import annotations

import pytest

from experiments.generic_utility.campaign import run_emulated_campaign


@pytest.fixture(scope="session")
def campaign_dir(tmp_path_factory):
    """Run the expensive campaign once; all assertions read the same receipt."""
    return run_emulated_campaign(tmp_path_factory.mktemp("generic-utility") / "campaign")
