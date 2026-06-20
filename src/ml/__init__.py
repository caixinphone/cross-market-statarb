"""Machine-learning enhancement (bonus / Task 6).

A two-layer *meta-learning* (stacking) sleeve on top of the baseline:

* **Layer 1 — timing (base learner):** the existing mean-reversion z-score state
  machine (:mod:`src.signals.zscore`) decides *when* to enter/exit and the sign.
* **Layer 2 — sizing (meta learner):** a 1D CNN (:mod:`src.ml.cnn`) looks at a
  point-in-time feature window at each entry and predicts the trade's
  **net-of-cost margin**; that prediction sets the trade *size* (predicted loss →
  0, predicted gain → scaled up).

Trained walk-forward (no lookahead) on a label that is gross idiosyncratic margin
minus fees/slippage/funding/borrow — so the CNN learns to allocate capital only to
trades that beat their costs, directly attacking the project's central finding.

This sleeve is OFF by default (``ml.enabled: false``); the baseline pipeline is
unchanged. Everything here is driven by :func:`scripts/run_ml.py`.
"""
