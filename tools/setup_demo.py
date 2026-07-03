"""One-command demo setup: dirs, dependency check, DB init, smoke test.

    python -m tools.setup_demo

Creates runtime directories, verifies the store imports and initializes,
runs a fast end-to-end smoke (build -> review -> replay on 2 sellers), and
prints the next command. Idempotent; safe to re-run.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

RUNTIME_DIRS = ["log", "log/operator_demo", "artifacts"]


def _check_deps() -> None:
    missing = []
    for mod in ("sqlite3", "http.server", "csv", "json", "unicodedata"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        raise SystemExit(f"setup failed: missing stdlib modules {missing}")


def _smoke() -> bool:
    from core.ledger_store import LedgerStore
    from core.population import build_population
    from core.review import replay_ledger, run_review_session
    store = LedgerStore(":memory:")
    worlds = build_population(n=2, seed=1)
    r = run_review_session(store, worlds, seed="smoke")
    ok = (r["proposed_events"] > 0
          and all(replay_ledger(store, w.profile.seller_id)[1] for w in worlds)
          and store.schema_version() >= 1)
    return ok


def main() -> None:
    print("ScreenGhost demo setup")
    _check_deps()
    for d in RUNTIME_DIRS:
        os.makedirs(os.path.join(_ROOT, d), exist_ok=True)
    print(f"  runtime dirs ready: {', '.join(RUNTIME_DIRS)}")

    # initialize a demo DB so first run is warm
    from core.ledger_store import LedgerStore
    demo_db = os.path.join(_ROOT, "log", "operator_demo", "ledger.db")
    LedgerStore(demo_db).close()
    print(f"  demo store initialized: {demo_db}")

    print("  running smoke test...", end=" ")
    if not _smoke():
        raise SystemExit("smoke test FAILED")
    print("ok")

    print("\nsetup complete. Next:")
    print('  python examples/operator_demo.py --seed "op" --sellers 10')


if __name__ == "__main__":
    main()
