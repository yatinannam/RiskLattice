"""Graph engine for RiskLattice.

A typed, temporal-aware relationship graph over the synthetic transaction
dataset. The graph models *relationships* (USER/DEVICE/IP/PAYMENT_INSTRUMENT/
TRANSACTION/MERCHANT) and is evidence-oriented — it is **not** itself a fraud
verdict. High degree or shared infrastructure here is evidence, never an
automatic fraud label.
"""