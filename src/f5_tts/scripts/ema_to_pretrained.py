"""Turn a training checkpoint into an EMA-only ``pretrained_*.safetensors``.

Stage hand-off for multi-stage training: the trainer resumes a ``.pt`` with its
optimizer, scheduler and update counter (that is what a ``.pt`` carrying
``update`` means to ``Trainer.load_checkpoint``), so the next stage would
continue the previous LR curve. An EMA-only file is loaded as pretrained
weights instead: the EMA becomes both model and EMA, everything else is fresh.

    python -m f5_tts.scripts.ema_to_pretrained \
        --ckpt checkpoints/f5-svs/muse/model_last.pt \
        --out  checkpoints/f5-svs/finetune/pretrained_muse_last.safetensors
"""

import argparse
import os

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="training checkpoint (.pt) with ema_model_state_dict")
    ap.add_argument("--out", required=True, help="output .safetensors (name it pretrained_*)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    ema = {k: v.contiguous() for k, v in ckpt["ema_model_state_dict"].items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(ema, args.out)
    print(f"{args.ckpt} (update {ckpt.get('update', '?')}) -> {args.out}: {len(ema)} EMA tensors")


if __name__ == "__main__":
    main()
