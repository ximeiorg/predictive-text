import torch

ckpt = torch.load("output/small/best_model.pt", map_location="cpu", weights_only=False)
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())[:20]
    print(f"best_model.pt keys: {keys}")

    sd = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    total = sum(p.numel() for p in sd.values() if isinstance(p, torch.Tensor))
    print(f"Total parameters: {total:,}")

    for field in ["epoch", "val_loss", "train_loss"]:
        if field in ckpt:
            print(f"{field}: {ckpt[field]}")

    if "optimizer" in ckpt:
        print("Has optimizer state: yes")

# Also check checkpoints/epoch=2-val_loss=3.4206.pt
ckpt2 = torch.load("output/small/checkpoints/epoch=2-val_loss=3.4206.pt", map_location="cpu", weights_only=False)
if isinstance(ckpt2, dict):
    keys2 = list(ckpt2.keys())[:15]
    print(f"\nepoch=2 checkpoint keys: {keys2}")
    for field in ["epoch", "val_loss", "train_loss"]:
        if field in ckpt2:
            print(f"{field}: {ckpt2[field]}")
