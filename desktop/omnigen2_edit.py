"""OmniGen2 image edit with the EditScore7B-v1.1 LoRA, on the local GPU.

No arguments: open the edit page in the browser (local only).
With --image and --prompt: edit one image and save it.

Weights folder, part of the package (default: ``w`` next to the program, or env OMNIGEN2_WEIGHTS):
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

# The program never connects to a network; the weights come from the w folder in the package.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Triton kernels compile C at run time and need a compiler and CUDA headers; the package has neither.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")


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
    from diffusers import AutoencoderKL
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    import omnigen2.utils.import_utils as iu
    iu._triton_available = False
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    base = weights / "base"
    lora = weights / "lora" / "pytorch_lora_weights.safetensors"
    if not (base / "model_index.json").is_file():
        sys.exit(f"Weights not found: {base}/model_index.json. The package holds them in the folder w.")
    dtype = torch.bfloat16
    t0 = time.time()
    # The two custom components come from the bundled omnigen2 package, never from Python files
    # inside the weights folder (no trust_remote_code), so the weights folder runs no code.
    transformer = OmniGen2Transformer2DModel.from_pretrained(str(base), subfolder="transformer", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(base), subfolder="scheduler")
    pipe = OmniGen2Pipeline(
        transformer=transformer,
        vae=AutoencoderKL.from_pretrained(str(base), subfolder="vae", torch_dtype=dtype),
        scheduler=scheduler,
        mllm=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(base), subfolder="mllm", torch_dtype=dtype),
        processor=Qwen2_5_VLProcessor.from_pretrained(str(base), subfolder="processor"),
    )
    if use_lora:
        if not lora.is_file():
            sys.exit(f"LoRA not found: {lora}")
        pipe.load_lora_weights(str(lora.parent), weight_name=lora.name)
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

    # System fonts only: the default theme fetches web fonts, and the page must stay offline.
    theme = gr.themes.Default(font=["ui-sans-serif", "system-ui", "sans-serif"],
                              font_mono=["ui-monospace", "monospace"])
    with gr.Blocks(title="OmniGen2 EditScore", theme=theme, analytics_enabled=False) as demo:
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
    demo.queue(max_size=4)
    serve_offline(demo, port)


def serve_offline(demo, port):
    """Serve the page on 127.0.0.1 only, with every script tag that loads from another computer removed
    (Gradio's page template loads an iframe helper from a CDN)."""
    import re
    import threading
    import webbrowser

    import gradio as gr
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    remote_script = re.compile(rb'<script[^>]*\bsrc="https?://[^"]*"[^>]*>\s*</script>')
    app = FastAPI()

    @app.middleware("http")
    async def strip_remote_scripts(request: Request, call_next):
        resp = await call_next(request)
        if not resp.headers.get("content-type", "").startswith("text/html"):
            return resp
        body = remote_script.sub(b"", b"".join([c async for c in resp.body_iterator]))
        headers = {k: v for k, v in resp.headers.items() if k.lower() != "content-length"}
        return Response(body, status_code=resp.status_code, headers=headers)

    app = gr.mount_gradio_app(app, demo, path="/")
    url = f"http://127.0.0.1:{port}/"
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    print(f"PAGE {url}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


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
