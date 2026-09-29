"""Grow a pretrained F5-TTS checkpoint's text embedding to an extended vocab.

For the VocalRender NAR ablation: the SVS vocab is F5's pretrained vocab with
the score tokens (<P_*>, <NOTE_*>, <BPM_*>, <SP>) appended. Existing rows are
copied, so every pretrained token keeps its id and its embedding; appended rows
are drawn from N(mean, std) of the trained rows, so they enter the ConvNeXt
text encoder at the scale it was trained on. (``finetune_gradio``'s
``expand_model_embeddings`` draws N(0, 1), ~1.6x the trained row norm.)

    python -m f5_tts.scripts.expand_vocab_svs \
        --ckpt  pretrained_models/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors \
        --base_vocab pretrained_models/F5-TTS/F5TTS_v1_Base/vocab.txt \
        --new_vocab  /dataset/cyk/F5-SVS/muse/vocab.txt \
        --out pretrained_models/F5-TTS/F5TTS_v1_Base_svs/pretrained_model_1250000_svs.safetensors

The output is EMA-only, like the released checkpoint, so the trainer loads it
as a ``pretrained_*`` file: fresh optimizer, scheduler and update counter.
"""

import argparse
import os

import torch
from safetensors.torch import load_file, save_file


EMBED_KEY = "ema_model.transformer.text_embed.text_embed.weight"


def read_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line[:-1] if line.endswith("\n") else line for line in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="pretrained EMA checkpoint (.safetensors)")
    ap.add_argument("--base_vocab", required=True, help="vocab.txt the checkpoint was trained with")
    ap.add_argument("--new_vocab", required=True, help="extended vocab.txt (base + appended tokens)")
    ap.add_argument("--out", required=True, help="output .safetensors (name it pretrained_*)")
    ap.add_argument("--seed", type=int, default=666)
    args = ap.parse_args()

    base, new = read_vocab(args.base_vocab), read_vocab(args.new_vocab)
    if new[: len(base)] != base:
        raise SystemExit("new vocab must start with the base vocab unchanged (ids would shift)")
    n_add = len(new) - len(base)

    sd = load_file(args.ckpt, device="cpu")
    old = sd[EMBED_KEY]
    # row 0 is the filler token; rows 1.. are vocab ids + 1
    if old.shape[0] != len(base) + 1:
        raise SystemExit(f"embedding has {old.shape[0]} rows, base vocab implies {len(base) + 1}")

    trained = old[1:].float()
    gen = torch.Generator().manual_seed(args.seed)
    extra = torch.randn((n_add, old.shape[1]), generator=gen) * trained.std() + trained.mean(0)
    sd[EMBED_KEY] = torch.cat([old, extra.to(old.dtype)], dim=0).contiguous()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(sd, args.out)
    print(
        f"{EMBED_KEY}: {tuple(old.shape)} -> {tuple(sd[EMBED_KEY].shape)} "
        f"(+{n_add} rows, std {trained.std():.3f}); wrote {args.out}"
    )


if __name__ == "__main__":
    main()
