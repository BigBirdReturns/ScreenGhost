PYTHON ?= python
SEED ?= op
SELLERS ?= 10

.PHONY: setup demo serve replay verify test smoke help

help:
	@echo "make setup   - dirs, deps, init DB, smoke test"
	@echo "make demo    - run the operator demo (SEED=op SELLERS=10)"
	@echo "make serve   - run the demo and open the local review UI"
	@echo "make replay  - replay the last demo store"
	@echo "make verify  - reproduce the canonical demo receipt"
	@echo "make test    - full test suite"

setup:
	$(PYTHON) -m tools.setup_demo

demo:
	$(PYTHON) examples/operator_demo.py --seed "$(SEED)" --sellers $(SELLERS)

serve:
	$(PYTHON) examples/operator_demo.py --seed "$(SEED)" --sellers $(SELLERS) --serve

replay:
	$(PYTHON) examples/replay_ledger.py --store log/operator_demo/ledger.db

verify:
	$(PYTHON) examples/verify_demo_receipt.py --receipt examples/receipts/operator_demo_seed_op.txt

smoke:
	$(PYTHON) -m tools.setup_demo

test:
	$(PYTHON) -m pytest tests/ -q
