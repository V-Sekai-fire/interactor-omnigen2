OmniGen2 + EditScore7B-v1.1
===========================

This program edits an image from an instruction, on your GPU. It uses OmniGen2
with the EditScore7B-v1.1 LoRA (OmniGen2 after reinforcement learning with the
EditScore-7B reward model).

1. Download the weights file omnigen2-editscore-weights.tar (31 GB) one time.
   Extract it in this folder. The result is the folder "w" (w/base and w/lora).
2. Start the program:
     macOS, Linux:  ./omnigen2-editscore
     Windows:       omnigen2-editscore.cmd
   The first start unpacks the Python environment into the folder "env".
   The program then opens the edit page in your browser. The page and the edit
   stay on this computer. The program does not download files.
3. Edit one image from a terminal:
     ./omnigen2-editscore --image in.png --prompt "Change the background to a classroom." --out edit.png
   Use --help to see all the options.

GPU: NVIDIA (CUDA 12) on Windows and Linux, Apple silicon (Metal) on macOS.
The model uses approximately 18 GB of GPU memory in BF16.

Licenses: OmniGen2 code and weights, Apache License 2.0 (LICENSE).
EditScore7B-v1.1 LoRA by VectorSpaceLab, Apache License 2.0.
