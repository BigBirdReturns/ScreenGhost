#!/usr/bin/env python3
import json
from experiments.emulator_fleet.doctor import doctor_report
print(json.dumps(doctor_report(), indent=2, sort_keys=True))
