import json, numpy as np, torch
import onnxruntime as ort
from src.model.lightning_module import DecoderTransformerLightningModule

with open("mobile/small/vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)
id2t = {v: k for k, v in vocab.items()}

pl = DecoderTransformerLightningModule.load_from_checkpoint(
    "output/small/best_model.pt", map_location="cpu"
)
model = pl.model.eval()

for tag, path in [("float32 (PyTorch)", None), ("int8 (ONNX)", "mobile/small/model_int8_dynamic.onnx")]:
    if tag == "int8 (ONNX)":
        sess = ort.InferenceSession(path)
        inp_name = sess.get_inputs()[0].name

    print(f"\n=== {tag} ===")
    for prefix in ["你好", "今天天气", "永远也看不", "我们一起去", "我在北京","梅花香自"]:
        ids = [vocab.get("[BOS]", 1)]
        for ch in prefix:
            ids.append(vocab.get(ch, vocab.get("[UNK]", 3)))
        inp = torch.tensor([ids])

        if tag == "float32 (PyTorch)":
            with torch.no_grad():
                logits = model(inp)["logits"][0, -1].numpy()
        else:
            logits = sess.run(None, {inp_name: inp.numpy()})[0][0, -1]

        top5 = np.argsort(logits)[-5:][::-1]
        words = [f"{id2t.get(int(i), '?')}(id={i})" for i in top5]
        print(f"  [{prefix}] {' | '.join(words)}")

# Compare directly
print("\n=== Top-5 token ID compare ===")
sess = ort.InferenceSession("mobile/small/model_int8_dynamic.onnx")
inp_name = sess.get_inputs()[0].name
total_match = 0
total = 0
for prefix in ["你好", "今天天气", "我觉得", "我们一起去", "我在北京"]:
    ids = [vocab.get("[BOS]", 1)]
    for ch in prefix:
        ids.append(vocab.get(ch, vocab.get("[UNK]", 3)))
    inp = torch.tensor([ids])
    with torch.no_grad():
        pt = model(inp)["logits"][0, -1].numpy()
    on = sess.run(None, {inp_name: inp.numpy()})[0][0, -1]
    pt_top5 = set(np.argsort(pt)[-5:][::-1])
    on_top5 = set(np.argsort(on)[-5:][::-1])
    match = pt_top5 & on_top5
    print(f"  [{prefix}] match: {len(match)}/5 -> IDs: {sorted(match)}")
    total_match += len(match)
    total += 5
print(f"\nTotal match rate: {total_match}/{total} = {total_match/total*100:.0f}%")
