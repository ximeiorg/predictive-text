from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

loader = EventFileLoader(
    "output/small/logs/lightning_logs/version_4/events.out.tfevents.1777909027.kingzcheung.11592.0"
)

count = 0
for event in loader.Load():
    for v in event.summary.value:
        if count < 30:
            has_tensor = v.HasField("tensor")
            print(f"step={event.step}, tag={v.tag}, val={v.simple_value}, has_tensor={has_tensor}")
            count += 1
        else:
            break
    if count >= 30:
        break

# Check variance in non-zero tags
print("\n--- Checking if val values are truly zero ---")
loader2 = EventFileLoader(
    "output/small/logs/lightning_logs/version_4/events.out.tfevents.1777909027.kingzcheung.11592.0"
)
totals = {}
for event in loader2.Load():
    for v in event.summary.value:
        if v.tag not in ("_hparams_/experiment", "_hparams_/session_end_info", "_hparams_/session_start_info"):
            if v.simple_value != 0.0:
                totals.setdefault(v.tag, []).append((event.step, v.simple_value))

if totals:
    print("Non-zero values found:")
    for tag, vals in sorted(totals.items()):
        print(f"  {tag}: {len(vals)} non-zero (first: {vals[0]})")
else:
    print("ALL values are 0.0 for non-hparams tags")
