#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from optim_utils import *
from svdiff.utils import save_unet_svdiff, save_text_encoder_svdiff

import yaml
from get_dataset_mimic_cxr import MimicCXRDataset
from imagenette_dataset import ImagenetteDataset
from tuxemon_dataset import TuxemonDataset
from adaptors import *
from parse_args import parse_args


if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.19.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def save_model_card(
    args,
    repo_id: str,
    images=None,
    repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
datasets:
- {args.dataset}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def log_validation(
    vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch
):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(
                args.validation_prompts[i], num_inference_steps=20, generator=generator
            ).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                "validation", np_images, epoch, dataformats="NHWC"
            )
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # if args.dataset is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    args.output_dir = os.path.join(args.output_dir, args.unet_pretraining_type)

    if args.use_random_word_addition:
        # args.output_dir = os.path.join(args.output_dir, args.unet_pretraining_type + "_RWA")
        args.output_dir = args.output_dir + "_RWA"
    if args.mitigation_threshold is not None:
        # args.output_dir = os.path.join(args.output_dir, args.unet_pretraining_type + "_Mitigation_{}".format(args.mitigation_threshold))
        args.output_dir = args.output_dir + "_Mitigation_{}".format(
            args.mitigation_threshold
        )

    # Import CSV path from the YAML file
    with open("data_config.yaml") as file:
        yaml_data = yaml.safe_load(file)

    args.train_data_path = yaml_data[args.dataset]["train_csv"]
    args.val_data_path = yaml_data[args.dataset]["val_csv"]

    if args.dataset == "MIMIC":
        args.test_data_path = yaml_data[args.dataset]["test_csv"]

    args.images_path_train = Path(yaml_data[args.dataset]["images_path_train"])
    args.images_path_val = Path(yaml_data[args.dataset]["images_path_val"])

    print("Train Data CSV: {}".format(args.train_data_path))

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        print("OUTPUT DIR: ", args.output_dir)
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        cache_dir=args.cache_dir,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin
            if accelerate.state.is_initialized()
            else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            cache_dir=args.cache_dir,
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.non_ema_revision,
        cache_dir=args.cache_dir,
    )

    # Load the Binary Mask here
    if args.binary_mask_path is not None:
        binary_mask = np.load(args.binary_mask_path)
        print("Binary Mask: ", binary_mask)

    # TODO: Add the UNet PEFT Logic here
    if args.unet_pretraining_type == "lorav2":
        raise NotImplementedError("LoRA v2 is not implemented yet.")

    # Normal SV-DIFF
    elif args.unet_pretraining_type == "svdiff":
        unet, optim_params, optim_params_1d = get_adapted_unet(
            unet=unet,
            method=args.unet_pretraining_type,
            args=args,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            cache_dir=args.cache_dir,
        )
    elif args.unet_pretraining_type == "auto_svdiff":

        assert binary_mask is not None
        assert len(binary_mask) == 13

        # Apply SV-DIFF to U-Net
        unet_with_svdiff, optim_params, optim_params_1d = get_adapted_unet(
            unet=unet,
            method="svdiff",
            args=args,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            cache_dir=args.cache_dir,
        )

        # Apply Mask to SV-DIFF U-Net
        unet, optim_params, optim_params_1d = enable_disable_svdiff_with_mask(
            unet_with_svdiff, binary_mask
        )

    elif args.unet_pretraining_type == "auto_difffit":

        assert binary_mask is not None
        assert len(binary_mask) == 13

        # Apply DiffFit to U-Net
        unet_with_difffit = get_adapted_unet(
            unet=unet,
            method="difffit",
            args=args,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            cache_dir=args.cache_dir,
        )

        # Apply Mask to DiffFit U-Net
        unet = enable_disable_difffit_with_mask(
            unet_with_difffit, binary_mask, verbose=False
        )

    elif args.unet_pretraining_type == "auto_attention":

        assert binary_mask is not None
        assert len(binary_mask) == 16

        unet = get_adapted_unet(
            unet=unet,
            method="attention",
            args=args,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        )

        # Apply Mask to Attention U-Net
        unet = enable_disable_attention_with_mask(unet, binary_mask)

    else:
        # Full FT, N, B, A, DiffFit
        unet = get_adapted_unet(
            unet=unet,
            method=args.unet_pretraining_type,
            args=args,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            cache_dir=args.cache_dir,
        )
    print("UNET")
    tunable_params = check_tunable_params(unet, False)

    if args.mitigation_threshold is not None:
        print("Using ICLR'24 Mitigation method")

    if args.use_random_word_addition:
        print("Using Random Word Addition")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if (
        args.unet_pretraining_type == "svdiff"
        or args.unet_pretraining_type == "auto_svdiff"
    ):
        optimizer = optimizer_cls(
            [
                {"params": optim_params},
                {"params": optim_params_1d, "lr": args.learning_rate_1d},
            ],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            (
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution)
            ),
            (
                transforms.RandomHorizontalFlip()
                if args.random_flip
                else transforms.Lambda(lambda x: x)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if args.dataset == "MIMIC":
        train_dataset = MimicCXRDataset(
            csv_file=args.train_data_path,
            images_dir=args.images_path_train,
            tokenizer=tokenizer,
            transform=train_transforms,
            seed=args.dataset_split_seed,
            dataset_size_ratio=args.data_size_ratio,
            use_real_images=True,
            use_random_word_addition=args.use_random_word_addition,
        )
        val_dataset = MimicCXRDataset(
            csv_file=args.val_data_path,
            images_dir=args.images_path_val,
            tokenizer=tokenizer,
            transform=val_transforms,
            seed=args.dataset_split_seed,
        )
        test_dataset = MimicCXRDataset(
            csv_file=args.test_data_path,
            images_dir=args.images_path_val,
            tokenizer=tokenizer,
            transform=val_transforms,
            seed=args.dataset_split_seed,
        )
    elif args.dataset == "imagenette":
        train_dataset = ImagenetteDataset(
            csv_file=args.train_data_path,
            root_path=args.images_path_train,
            tokenizer=tokenizer,
            transform=train_transforms,
            use_random_word_addition=args.use_random_word_addition,
        )
        val_dataset = ImagenetteDataset(
            csv_file=args.val_data_path,
            root_path=args.images_path_val,
            tokenizer=tokenizer,
            transform=val_transforms,
        )
    elif args.dataset == "tuxemon":
        train_dataset = TuxemonDataset(
            csv_file=args.train_data_path,
            root_path=args.images_path_train,
            tokenizer=tokenizer,
            transform=train_transforms,
            use_random_word_addition=args.use_random_word_addition,
        )
        val_dataset = TuxemonDataset(
            csv_file=args.val_data_path,
            root_path=args.images_path_val,
            tokenizer=tokenizer,
            transform=val_transforms,
        )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    if args.dataset == "MIMIC":
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples (train) = {len(train_dataset)}")
    logger.info(f"  Num examples (val) = {len(val_dataset)}")
    # logger.info(f"  Num examples (test) = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    if not args.disable_training:
        for epoch in range(first_epoch, args.num_train_epochs):
            unet.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if (
                    args.resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(
                        # batch["pixel_values"].to(weight_dtype)
                        batch["image"].to(weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1),
                            device=latents.device,
                        )
                    if args.input_perturbation:
                        new_noise = noise + args.input_perturbation * torch.randn_like(
                            noise
                        )
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if args.input_perturbation:
                        noisy_latents = noise_scheduler.add_noise(
                            latents, new_noise, timesteps
                        )
                    else:
                        noisy_latents = noise_scheduler.add_noise(
                            latents, noise, timesteps
                        )

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Get the target for loss depending on the prediction type
                    if args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(
                            prediction_type=args.prediction_type
                        )

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                    # Predict the noise residual and compute loss
                    model_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    ### mitigation:
                    # NOTE: Don't think we need to perform mitigation here
                    if args.mitigation_threshold is not None:
                        with torch.no_grad():
                            uncond_tokens = [""] * len(model_pred)
                            uncond_input = tokenizer(
                                uncond_tokens,
                                padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt",
                            )
                            uncond_prompt_embed = text_encoder(
                                uncond_input["input_ids"].to(accelerator.device)
                            )[0]
                            noise_pred_uncond = unet(
                                noisy_latents, timesteps, uncond_prompt_embed
                            ).sample
                            noise_pred_text = model_pred - noise_pred_uncond
                            noise_pred_text = noise_pred_text.reshape(
                                len(noise_pred_text), -1
                            )
                            noise_pred_text_norm = noise_pred_text.norm(p=2, dim=1)

                        # Answer: Remove the samples that have a memorization detection metric above the threshold

                        # import pdb; pdb.set_trace()

                        if args.mitigation_threshold == "auto":
                            # Remove the samples with top 10% memorization detection metric
                            auto_threshold = torch.quantile(noise_pred_text_norm, 0.90)

                            model_pred = model_pred[
                                noise_pred_text_norm < auto_threshold
                            ]
                            target = target[noise_pred_text_norm < auto_threshold]
                        else:
                            args.mitigation_threshold = float(args.mitigation_threshold)

                            model_pred = model_pred[
                                noise_pred_text_norm < args.mitigation_threshold
                            ]
                            target = target[
                                noise_pred_text_norm < args.mitigation_threshold
                            ]

                    if len(model_pred) != 0:
                        if args.snr_gamma is None:
                            loss = F.mse_loss(
                                model_pred.float(), target.float(), reduction="mean"
                            )
                        else:
                            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                            # This is discussed in Section 4.2 of the same paper.
                            snr = compute_snr(timesteps)
                            mse_loss_weights = (
                                torch.stack(
                                    [snr, args.snr_gamma * torch.ones_like(timesteps)],
                                    dim=1,
                                ).min(dim=1)[0]
                                / snr
                            )
                            # We first calculate the original loss. Then we mean over the non-batch dimensions and
                            # rebalance the sample-wise losses with their respective loss weights.
                            # Finally, we take the mean of the rebalanced loss.
                            loss = F.mse_loss(
                                model_pred.float(), target.float(), reduction="none"
                            )
                            loss = (
                                loss.mean(dim=list(range(1, len(loss.shape))))
                                * mse_loss_weights
                            )
                            loss = loss.mean()

                        # Gather the losses across all processes for logging (if we use distributed training).
                        avg_loss = accelerator.gather(
                            loss.repeat(args.train_batch_size)
                        ).mean()
                        train_loss += avg_loss.item() / args.gradient_accumulation_steps

                        # print(loss)
                        # Backpropagate
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                unet.parameters(), args.max_grad_norm
                            )
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [
                                    d for d in checkpoints if d.startswith("checkpoint")
                                ]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = (
                                        len(checkpoints)
                                        - args.checkpoints_total_limit
                                        + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            args.output_dir, removing_checkpoint
                                        )
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(
                                args.output_dir, f"checkpoint-{global_step}"
                            )
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

        # Create the pipeline using the trained modules and save it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(unet)
            if args.use_ema:
                ema_unet.copy_to(unet.parameters())

            # NOTE: Save pipeline according to the PEFT type
            if (
                args.unet_pretraining_type == "svdiff"
                or args.unet_pretraining_type == "auto_svdiff"
            ):
                svdiff_unet_savedir = os.path.join(args.output_dir, "unet")

                if not os.path.exists(svdiff_unet_savedir):
                    os.makedirs(svdiff_unet_savedir)

                # Save UNet
                save_unet_svdiff(accelerator, unet, svdiff_unet_savedir)

                # Save Text Encoder and VAE the normal way
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    revision=args.revision,
                    cache_dir=args.cache_dir,
                )
                pipeline.save_pretrained(args.output_dir)

            else:
                print("SAVING THE SD PIPELINE AT {}".format(args.output_dir))

                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    revision=args.revision,
                    cache_dir=args.cache_dir,
                )
                pipeline.save_pretrained(args.output_dir)

        accelerator.end_training()
    else:
        print("Training is disabled. Exiting the training loop.")


if __name__ == "__main__":
    main()
