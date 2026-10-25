from verigate_verifier.config import load
from verigate_verifier.store import Store
from verigate_verifier.stats import StatsEngine

cfg = load('verigate.yaml')
store = Store(cfg.sqlite_path)
engine = StatsEngine(store, cfg.stats, active_pairs=len(cfg.providers))

for p in ('lab-honest', 'lab-noisy'):
    res = engine.evaluate_provider(p, 'qwen2.5-0.5b-instruct')
    print(f"Provider {p}: state={res.state}, n_audits={res.n_audits}, combined_p={res.combined_p:.5f}")
    for t in res.test_results:
        w_m = t.detail.get("median_window")
        n_m = t.detail.get("median_null")
        w_str = f"{w_m:.3f}" if isinstance(w_m, (int, float)) else "N/A"
        n_str = f"{n_m:.3f}" if isinstance(n_m, (int, float)) else "N/A"
        print(f"   {t.signal} {t.test_type}: p={t.p_value:.5f}, w_median={w_str}, null_med={n_str}")
