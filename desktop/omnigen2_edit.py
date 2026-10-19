"""OmniGen2 image edit with the EditScore7B-v1.1 LoRA, on the local GPU.

No arguments: open the edit page in the browser (local only).
With --image and --prompt: edit one image and save it.

Weights folder (default: ``w`` next to the program, or env OMNIGEN2_WEIGHTS):
  w/base  OmniGen2 base model (diffusers layout)
  w/lora  pytorch_lora_weights.safetensors (OmniGen2-EditScore7B-v1.1)
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The program never downloads anything; all weights come from the w folder.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def default_weights() -> Path:
    env = os.environ.get("OMNIGEN2_WEIGHTS")
    if env:
        return Path(env)
    for cand in (HERE.parent / "w", HERE / "w"):
        if cand.is_dir():
            return cand
    return HERE.parent / "w"


def pick_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(weights: Path, device, use_lora: bool = True):
    import torch
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

    base = weights / "base"
    lora = weights / "lora" / "pytorch_lora_weights.safetensors"
    if not (base / "model_index.json").is_file():
        sys.exit(f"Weights not found: {base}/model_index.json. Unpack the weights archive into {weights}.")
    dtype = torch.bfloat16
    t0 = time.time()
    pipe = OmniGen2Pipeline.from_pretrained(str(base), torch_dtype=dtype, trust_remote_code=True)
    pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(str(base), subfolder="transformer", torch_dtype=dtype)
    if use_lora:
        if not lora.is_file():
            sys.exit(f"LoRA not found: {lora}")
        pipe.load_lora_weights(str(lora))
    pipe = pipe.to(device)
    print(f"LOADED device={device} lora={use_lora} seconds={time.time() - t0:.1f}", flush=True)
    return pipe


def edit(pipe, device, image, prompt, steps=50, seed=0, text_cfg=5.0, image_cfg=2.0, max_side=1024,
         negative="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, "
                  "mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, "
                  "broken legs censor, censored, censor_bar"):
    import torch
    from PIL import ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    w, h = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
    gen = torch.Generator(device="cpu" if device.type == "mps" else device).manual_seed(int(seed))
    out = pipe(
        prompt=prompt,
        input_images=[image],
        width=w,
        height=h,
        num_inference_steps=int(steps),
        max_sequence_length=1024,
        text_guidance_scale=float(text_cfg),
        image_guidance_scale=float(image_cfg),
        cfg_range=(0.0, 1.0),
        negative_prompt=negative,
        num_images_per_prompt=1,
        generator=gen,
        output_type="pil",
    )
    return out.images[0]


def run_ui(pipe, device, port):
    import gradio as gr

    def go(image, prompt, steps, seed):
        if image is None or not prompt:
            raise gr.Error("Give an image and an instruction.")
        t0 = time.time()
        res = edit(pipe, device, image, prompt, steps=steps, seed=seed)
        return res, f"{time.time() - t0:.0f} s on {device}"

    with gr.Blocks(title="OmniGen2 EditScore") as demo:
        gr.Markdown("## OmniGen2 + EditScore7B-v1.1\nGive an image and an instruction. The edit runs on this computer.")
        with gr.Row():
            src = gr.Image(type="pil", label="Image")
            dst = gr.Image(type="pil", label="Edit", format="png")
        prompt = gr.Textbox(label="Instruction", placeholder="Change the background to a classroom.")
        with gr.Row():
            steps = gr.Slider(10, 50, value=50, step=1, label="Steps")
            seed = gr.Number(value=0, precision=0, label="Seed")
        btn = gr.Button("Edit", variant="primary")
        info = gr.Markdown()
        btn.click(go, [src, prompt, steps, seed], [dst, info])
    demo.queue(max_size=4).launch(server_name="127.0.0.1", server_port=port, inbrowser=True, share=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="input image")
    ap.add_argument("--prompt", help="edit instruction")
    ap.add_argument("--out", default="edit.png", help="output PNG (default edit.png)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1024, help="longest output side in pixels")
    ap.add_argument("--weights", type=Path, default=default_weights())
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--no-lora", action="store_true", help="base OmniGen2 only (for comparison)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()

    device = pick_device(a.device)
    pipe = load(a.weights, device, use_lora=not a.no_lora)
    if a.image or a.prompt:
        if not (a.image and a.prompt):
            sys.exit("--image and --prompt go together")
        from PIL import Image

        t0 = time.time()
        res = edit(pipe, device, Image.open(a.image), a.prompt, steps=a.steps, seed=a.seed, max_side=a.max_side)
        res.save(a.out)
        print(f"EDIT-OK {a.out} {res.size[0]}x{res.size[1]} seconds={time.time() - t0:.1f}", flush=True)
        return
    run_ui(pipe, device, a.port)


if __name__ == "__main__":
    main()
