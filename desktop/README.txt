OmniGen2 + EditScore7B-v1.1
===========================

This program edits an image from an instruction, on your GPU. It uses OmniGen2
with the EditScore7B-v1.1 LoRA (OmniGen2 after reinforcement learning with the
EditScore-7B reward model). The LoRA is merged into the weights, and the weights
are stored in 4 bits (4.6 GB).

1. Start the program:
     macOS, Linux:  ./omnigen2-editscore
     Windows:       omnigen2-editscore.cmd
   The first start unpacks the Python environment into the folder "env".
   The weights are in the folder "w" of this package. The program does not
   connect to other computers. It opens the edit page in your browser; the page
   and the edit stay on this computer.
2. Edit one image from a terminal:
     ./omnigen2-editscore --image in.png --prompt "Change the background to a classroom." --out edit.png
   Use --help to see all the options.

GPU: NVIDIA (CUDA 12) on Windows and Linux, Apple silicon (Metal) on macOS.
On NVIDIA, the program expands the weights to BF16 when it starts. It uses
approximately 17 GB of GPU memory. On Apple silicon, the weights stay in 4 bits.

Licenses: OmniGen2 code and weights, Apache License 2.0 (LICENSE).
EditScore7B-v1.1 LoRA by VectorSpaceLab, Apache License 2.0.
