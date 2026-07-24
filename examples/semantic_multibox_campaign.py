#!/usr/bin/env python3
from pathlib import Path
import argparse
from experiments.emulator_fleet.campaign import run_semantic_multibox_campaign

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, default=Path("log/semantic_multibox/campaign"))
args = p.parse_args()
print(run_semantic_multibox_campaign(args.out))
