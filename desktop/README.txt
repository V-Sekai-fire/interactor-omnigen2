OmniGen2 + EditScore7B-v1.1
===========================

This program edits an image from an instruction, on your GPU. It uses OmniGen2
with the EditScore7B-v1.1 LoRA (OmniGen2 after reinforcement learning with the
EditScore-7B reward model).

1. Get the weights one time (31 GB, from Hugging Face):
     macOS, Linux:  ./omnigen2-editscore --download-weights
     Windows:       omnigen2-editscore.cmd --download-weights
   The first start unpacks the Python environment into the folder "env".
   The weights go into the folder "w" (w/base and w/lora), at fixed revisions.
2. Start the program:
     macOS, Linux:  ./omnigen2-editscore
     Windows:       omnigen2-editscore.cmd
   The program opens the edit page in your browser. The page and the edit
   stay on this computer. Only step 1 connects to other computers.
3. Edit one image from a terminal:
     ./omnigen2-editscore --image in.png --prompt "Change the background to a classroom." --out edit.png
   Use --help to see all the options.

GPU: NVIDIA (CUDA 12) on Windows and Linux, Apple silicon (Metal) on macOS.
The model uses approximately 18 GB of GPU memory in BF16.

Licenses: OmniGen2 code and weights, Apache License 2.0 (LICENSE).
EditScore7B-v1.1 LoRA by VectorSpaceLab, Apache License 2.0.
