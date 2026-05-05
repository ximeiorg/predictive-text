from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator("output/small/logs/lightning_logs/version_3")
ea.Reload()

loss = ea.Scalars("train/loss_step")
print(f"Training loss ({len(loss)} steps):")
for i in range(0, len(loss), max(1, len(loss) // 10)):
    s = loss[i]
    print(f"  step={s.step:>6d}  loss={s.value:.4f}")
print(f"  final:  step={loss[-1].step:>6d}  loss={loss[-1].value:.4f}")

gn = ea.Scalars("train/grad_norm")
if gn:
    print(f"\nGradient norm ({len(gn)} records):")
    for i in range(0, len(gn), max(1, len(gn) // 8)):
        g = gn[i]
        print(f"  step={g.step:>6d}  grad_norm={g.value:.4f}")
    print(f"  final:  step={gn[-1].step:>6d}  grad_norm={gn[-1].value:.4f}")

val = ea.Scalars("val/loss")
if val:
    print(f"\nValidation loss:")
    for v in val:
        print(f"  step={v.step:>6d}  val/loss={v.value:.4f}")

best = ea.Scalars("val/best_loss") if "val/best_loss" in ea.Tags()["scalars"] else []
if best:
    print(f"  best:   val/loss={best[-1].value:.4f}")

lr = ea.Scalars("train/lr")
if lr:
    print(f"\nLR: first={lr[0].value:.6f}  last={lr[-1].value:.6f}")

ep = ea.Scalars("epoch")
if ep:
    print(f"Epoch: {ep[-1].value}")
