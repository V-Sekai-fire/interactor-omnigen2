"""4-bit weights for the transformer and the language model of OmniGen2.

Layout per quantized nn.Linear: groups of G consecutive input weights in one output row share one
BF16 scale and one BF16 minimum (asymmetric round to nearest, 16 levels). G is the largest of the
preferred sizes that divides the input width, so the 2520-wide layers need no padding. Codes are
packed two per byte.

A component folder that holds ``q4.json`` and ``model.q4.safetensors`` (written by make_q4.py)
loads here instead of from its BF16 files. On CUDA the packed weights are unpacked once at load to
BF16, so the edit runs at full BF16 speed. On Apple GPUs and the CPU they stay packed and each
layer unpacks its weights when it runs, so the weights take about a quarter of the memory.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

PREFERRED = (128, 120, 112, 105, 90, 84, 72, 70, 64, 63, 60, 56, 48, 45, 42, 40, 36, 35, 32)
FORMAT = "omnigen2-q4/1"


def group_for(width: int) -> int:
    for g in PREFERRED:
        if width % g == 0:
            return g
    raise ValueError(f"no group size divides {width}")


class QLinear(nn.Module):
    def __init__(self, out_features: int, in_features: int, g: int, bias: bool, device=None):
        super().__init__()
        self.out_features, self.in_features, self.g = out_features, in_features, g
        self.register_buffer("q", torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device))
        self.register_buffer("scale", torch.empty(out_features, in_features // g, 1, dtype=torch.bfloat16, device=device))
        self.register_buffer("mn", torch.empty(out_features, in_features // g, 1, dtype=torch.bfloat16, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device), requires_grad=False) if bias else None

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "QLinear":
        o, i = lin.weight.shape
        g = group_for(i)
        m = cls(o, i, g, lin.bias is not None, device=lin.weight.device)
        w = lin.weight.data.float().view(o, i // g, g)
        mn = w.amin(-1, keepdim=True)
        scale = ((w.amax(-1, keepdim=True) - mn) / 15).clamp_min(1e-12)
        # Round the scale and minimum to BF16 first, so the codes fit the values that are stored.
        scale, mn = scale.to(torch.bfloat16), mn.to(torch.bfloat16)
        q = ((w - mn.float()) / scale.float()).round().clamp(0, 15).to(torch.uint8).view(o, i)
        m.q.copy_(q[:, 0::2] | (q[:, 1::2] << 4))
        m.scale.copy_(scale)
        m.mn.copy_(mn)
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.data)
        return m

    def unpacked_weight(self) -> torch.Tensor:
        q = torch.stack((self.q & 15, self.q >> 4), -1).view(self.out_features, self.in_features // self.g, self.g)
        return (q.to(torch.bfloat16) * self.scale + self.mn).view(self.out_features, self.in_features)

    @property
    def weight(self):  # read by code that inspects .weight.dtype / .device
        return self.unpacked_weight()

    def forward(self, x):
        b = None if self.bias is None else self.bias.to(x.dtype)
        return nn.functional.linear(x, self.unpacked_weight().to(x.dtype), b)

    def to_linear(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                        device=self.q.device, dtype=torch.bfloat16)
        lin.weight.data.copy_(self.unpacked_weight())
        if self.bias is not None:
            lin.bias.data.copy_(self.bias.data)
        return lin


def _parent(root: nn.Module, dotted: str):
    *path, last = dotted.split(".")
    for k in path:
        root = getattr(root, k)
    return root, last


# Which layers are 4-bit: every Linear in the language model except its output head, and every
# Linear in the transformer blocks. Embedders, time and final layers of the transformer stay BF16.
TRANSFORMER_BLOCKS = ("layers", "noise_refiner", "ref_image_refiner", "context_refiner")


def chosen(model: nn.Module, part: str) -> list[str]:
    out = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if part == "mllm" and not n.endswith("lm_head"):
            out.append(n)
        elif part == "transformer" and n.split(".")[0] in TRANSFORMER_BLOCKS:
            out.append(n)
    return out


def quantize(model: nn.Module, part: str) -> list[str]:
    names = chosen(model, part)
    for n in names:
        p, k = _parent(model, n)
        setattr(p, k, QLinear.from_linear(getattr(p, k)))
    return names


def save(model: nn.Module, folder: Path, names: list[str], config_files=("config.json",)):
    from safetensors.torch import save_file
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {}
    persistent = {}
    seen = set()
    for mod_name, mod in model.named_modules():
        for k, t in list(mod._parameters.items()) + list(mod._buffers.items()):
            if t is None:
                continue
            full = f"{mod_name}.{k}" if mod_name else k
            if id(t) in seen:
                continue  # tied weight: stored once, re-tied after load
            seen.add(id(t))
            tensors[full] = t.detach().contiguous().cpu()
            if k in mod._buffers:
                persistent[full] = k not in mod._non_persistent_buffers_set
    save_file(tensors, str(folder / "model.q4.safetensors"))
    meta = {"format": FORMAT, "quantized": {n: getattr(_parent(model, n)[0], _parent(model, n)[1]).g for n in names},
            "buffers": persistent}
    (folder / "q4.json").write_text(json.dumps(meta, indent=0))


def is_q4(folder: Path) -> bool:
    return (folder / "q4.json").is_file()


def load(build_empty, folder: Path, device, unpack: bool) -> nn.Module:
    """build_empty() returns the model built on the meta device. Fills it from the q4 files."""
    from safetensors.torch import load_file
    meta = json.loads((folder / "q4.json").read_text())
    assert meta["format"] == FORMAT, meta["format"]
    model = build_empty()
    for n, g in meta["quantized"].items():
        p, k = _parent(model, n)
        lin = getattr(p, k)
        setattr(p, k, QLinear(lin.out_features, lin.in_features, g, lin.bias is not None, device="meta"))
    sd = load_file(str(folder / "model.q4.safetensors"), device=str(device) if device.type != "mps" else "cpu")
    for full, t in sd.items():
        p, k = _parent(model, full)
        if full in meta["buffers"]:
            p.register_buffer(k, t.to(device), persistent=meta["buffers"][full])
        else:
            setattr(p, k, nn.Parameter(t.to(device), requires_grad=False))
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    assert not left, f"weights missing from {folder}: {left[:5]}"
    if unpack:
        for n in meta["quantized"]:
            p, k = _parent(model, n)
            setattr(p, k, getattr(p, k).to_linear())
    return model.eval()
