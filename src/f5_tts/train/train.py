# training script.

import importlib
import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    wandb_project = model_cfg.ckpts.get("wandb_project", "CFM-TTS")
    wandb_run_name = model_cfg.ckpts.get(
        "wandb_run_name",
        f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}",
    )
    wandb_resume_id = model_cfg.ckpts.get("wandb_resume_id", None)

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    # init trainer
    # Built before the Trainer, which compiles the model: validators may copy it.
    validation, val_cfg = None, model_cfg.get("validation")
    if val_cfg is not None and val_cfg.get("target"):
        module_name, _, cls_name = val_cfg.target.rpartition(".")
        validation = getattr(importlib.import_module(module_name), cls_name)(
            model, **OmegaConf.to_container(val_cfg.get("kwargs", {}), resolve=True)
        )

    # Validation loss on a held-out F5 CustomDataset directory (raw.arrow + duration.json).
    val_loss_cfg = model_cfg.get("val_loss") or {}
    val_loss_dataset = None
    if val_loss_cfg.get("dataset_dir"):
        val_loss_dataset = load_dataset(
            val_loss_cfg.dataset_dir, dataset_type="CustomDatasetPath", mel_spec_kwargs=model_cfg.model.mel_spec
        )

    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        # an absolute save_dir is used as is; joinpath("../../" + "/abs") would nest it under the repo
        checkpoint_path=(
            model_cfg.ckpts.save_dir
            if os.path.isabs(model_cfg.ckpts.save_dir)
            else str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}"))
        ),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        compile=model_cfg.model.get("compile", False),
        validation=validation,
        validate_per_updates=val_cfg.get("every_updates", 0) if validation is not None else 0,
        validate_at_start=val_cfg.get("at_start", False) if validation is not None else False,
        val_loss_dataset=val_loss_dataset,
        val_loss_per_updates=val_loss_cfg.get("every_updates", 0),
        val_loss_at_start=val_loss_cfg.get("at_start", False),
        val_loss_seed=val_loss_cfg.get("seed", 0),
        val_loss_num_workers=val_loss_cfg.get("num_workers", 4),
    )

    train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec)
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
