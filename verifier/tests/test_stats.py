import numpy as np
import pytest

from verigate_verifier.config import StatsConfig
from verigate_verifier.stats import (
    STATE_INCONCLUSIVE,
    STATE_SUSPECT,
    STATE_VERIFIED,
    StatsEngine,
)
from verigate_verifier.store import Store


def test_verdict_state_machine_flow(tmp_path):
    db_file = tmp_path / "test.db"
    store = Store(db_file)
    cfg = StatsConfig(min_audits=10, min_per_signal=5, min_null_per_bucket=10)
    engine = StatsEngine(store, cfg)

    np.random.seed(42)
    provider = "test-prov"
    model = "test-model"

    # 1. Zero audits -> INCONCLUSIVE
    eval1 = engine.evaluate_provider(provider, model)
    assert eval1.state == STATE_INCONCLUSIVE
    assert eval1.n_audits == 0

    # Populate null distribution (empirical honest calibration, uniform grid around 0.0)
    null_vals = np.linspace(-2.0, 2.0, 50)
    null_rows = [
        {
            "model_alias": model,
            "signal": "s2",
            "temp_bucket": "t1",
            "len_bucket": "short",
            "value": float(v),
            "source": "calib",
            "created_at": "2026-10-21T00:00:00Z",
        }
        for v in null_vals
    ]
    store.insert_null_samples(null_rows)

    # 2. Insert 12 audits from honest distribution (median ~ 0.0) -> VERIFIED
    honest_vals = np.linspace(-1.5, 1.5, 12)
    for i, v in enumerate(honest_vals):
        store.record_audit({
            "audit_id": f"aud_h_{i}",
            "request_id": f"req_h_{i}",
            "provider": provider,
            "model_alias": model,
            "received_at": f"2026-10-21T00:{i:02d}:00Z",
            "status": "scored",
            "s2": float(v),
        })

    eval2 = engine.evaluate_provider(provider, model)
    assert eval2.n_audits >= 10
    assert eval2.state == STATE_VERIFIED

    # 3. Now flood with 20 extreme substitute audits (mean shift + extreme outliers) -> SUSPECT
    for i in range(20):
        store.record_audit({
            "audit_id": f"aud_bad_{i}",
            "request_id": f"req_bad_{i}",
            "provider": provider,
            "model_alias": model,
            "received_at": f"2026-10-21T01:{i:02d}:00Z",
            "status": "scored",
            "s2": 15.0 + float(np.random.normal(0.0, 0.5)),
        })

    eval3 = engine.evaluate_provider(provider, model)
    assert eval3.state == STATE_SUSPECT
    assert eval3.combined_p < 1e-4
