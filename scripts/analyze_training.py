from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from collections import defaultdict

loader = EventFileLoader(
    "output/small/logs/lightning_logs/version_4/events.out.tfevents.1777909027.kingzcheung.11592.0"
)

data = defaultdict(list)
for event in loader.Load():
    step = event.step
    for v in event.summary.value:
        if v.tag in (
            "train/loss_step", "train/loss_epoch", "train/lr",
            "train/grad_norm", "val/loss", "val/best_loss",
            "epoch", "hp_metric",
        ):
            data[v.tag].append((step, v.simple_value))

for tag, vals in sorted(data.items()):
    steps = [s for s, _ in vals]
    vals_f = [v for _, v in vals]
    print(f"\n=== {tag} ===")
    print(f"  Records: {len(vals)}")
    print(f"  Steps:   {min(steps)}-{max(steps)}")
    if tag in ("epoch", "hp_metric"):
        print(f"  Values:  {[round(v, 2) for v in vals_f[:10]]}...")
    elif tag == "train/lr":
        print(f"  First 5: {[f'{v:.8f}' for v in vals_f[:5]]}")
        print(f"  Last  5: {[f'{v:.8f}' for v in vals_f[-5:]]}")
        print(f"  Min/Max: {min(vals_f):.8f}/{max(vals_f):.8f}")
    else:
        print(f"  First:   {vals_f[0]:.4f}")
        print(f"  Last:    {vals_f[-1]:.4f}")
        print(f"  Min:     {min(vals_f):.4f}")
        print(f"  Max:     {max(vals_f):.4f}")
