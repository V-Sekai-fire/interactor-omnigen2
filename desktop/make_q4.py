"""Make the 4-bit weights folder from the BF16 weights folder.

Usage: make_q4.py SRC_W DST_W
SRC_W: w/base + w/lora (BF16, as in the BF16 packages). DST_W gets base/ with the LoRA merged into
the transformer, the transformer blocks and the language model in 4 bits (quant4.py), and every
other part copied unchanged. Runs on the CPU.
"""
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import omnigen2_edit as og  # first: it sets the environment before torch loads
import torch
import quant4

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists():
    sys.exit(f"{dst} exists")
t0 = time.time()
pipe = og.load(src, torch.device("cpu"), use_lora=True)
pipe.fuse_lora(lora_scale=1.0)
pipe.unload_lora_weights()
left = [n for n, _ in pipe.transformer.named_parameters() if "lora" in n.lower()]
assert not left, f"LoRA parameters left after the merge: {left[:3]}"

(dst / "base").mkdir(parents=True)
for item in (src / "base").iterdir():
    if item.name in ("transformer", "mllm"):
        continue
    (shutil.copytree if item.is_dir() else shutil.copy2)(item, dst / "base" / item.name)
for part, model in (("transformer", pipe.transformer), ("mllm", pipe.mllm)):
    names = quant4.quantize(model, part)
    folder = dst / "base" / part
    quant4.save(model, folder, names)
    for f in (src / "base" / part).iterdir():
        if f.suffix == ".json" and not f.name.endswith("index.json"):
            shutil.copy2(f, folder / f.name)
    size = (folder / "model.q4.safetensors").stat().st_size
    print(f"Q4 {part} layers={len(names)} file={size / 2**30:.2f} GiB", flush=True)
total = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
print(f"Q4-OK {dst} total={total / 2**30:.2f} GiB seconds={time.time() - t0:.0f}", flush=True)
