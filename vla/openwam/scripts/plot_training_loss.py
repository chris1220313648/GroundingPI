"""Plot recorded training loss without changing the training protocol."""
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
run = Path(sys.argv[1])
with (run / "debug_loss_history.csv").open() as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("No recorded training metrics")
print("Metric columns:", list(rows[0]))
xkey = "opt_step" if "opt_step" in rows[0] else "global_step"
ykey = "loss_action" if "loss_action" in rows[0] else "loss_total"
plt.plot([float(r[xkey]) for r in rows], [float(r[ykey]) for r in rows], linewidth=.6)
plt.xlabel(xkey)
plt.ylabel(ykey)
plt.tight_layout()
plt.savefig(run / "training_loss.png", dpi=160)
