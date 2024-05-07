import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import (
    LoraModel,
    LoraConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft import OFTModel, OFTConfig
from peft import LoKrModel, LoKrConfig
from peft import LoHaModel, LoHaConfig

from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from svdiff.utils import (
    load_unet_for_svdiff,
    load_text_encoder_for_svdiff,
    SCHEDULER_MAPPING,
)
from diffusers.loaders import AttnProcsLayers

from difffit.unet_2d_condition import UNet2DConditionModel as UNet2DConditionModelForDiffFit
from difffit.attention_processor import DiffFitAttnProcessor, DiffFitXFormersAttnProcessor
from difffit.modeling_clip import CLIPTextModel as CLIPTextModelForDiffFit

# from Scale_Shift_Features.unet_2d_condition import UNet2DConditionModel as UNet2DConditionModelForSSF
# from Scale_Shift_Features.attention_processor import SSFAttnProcessor, SSFXFormersAttnProcessor
# from Scale_Shift_Features.modeling_clip import CLIPTextModel as CLIPTextModelForDiffFit

from typing import Dict
import inspect
import accelerate
from accelerate.utils import set_module_tensor_to_device
from transformers import CLIPTextModel, CLIPTextConfig
from diffusers import UNet2DConditionModel
from safetensors.torch import safe_open
import huggingface_hub


def enable_grad(module):
    for p in module.parameters():
        p.requires_grad = True


def disable_grad(module):
    for p in module.parameters():
        p.requires_grad = False

def disable_grad_svdiff(module):
    for n,p in module.named_parameters():
        if("delta" in n):
            p.requires_grad = False

def disable_grad_difffit(module):
    for n,p in module.named_parameters():
        if("gamma_" in n):
            p.requires_grad = False

def disable_grad_attention(module):
    for n,p in module.named_parameters():
        if("attn" in n):
            p.requires_grad = False

def set_module_grad_status(module, flag=False):
    if isinstance(module, list):
        print("list", module)
        for m in module:
            set_module_grad_status(m, flag)
    else:
        print("not a list", module)
        for p in module.parameters():
            p.requires_grad = flag


def return_param_sum(model, verbose=False):
    total_magnitude = 0
    # print("Calculating sum of trainable parameters")
    
    for p in model.parameters():
        if p.requires_grad:
            total_magnitude += p.sum().item()

    return round(total_magnitude, 3)

def return_blockwise_param_sum(model, verbose=False):

    sum_dict_down_blocks = {}
    sum_dict_mid_block = {}
    sum_dict_up_blocks = {}

    # print("Calculating the sum of parameters in U-net Down Blocks")
    for idx, block in enumerate(model.down_blocks):
        sum_dict_down_blocks['ALL_Down_Block_'+str(idx)] = return_param_sum(block, False)

    # print("Calculating the sum of parameters in U-net Mid Block")
    sum_dict_mid_block['ALL_Mid_Block'] = return_param_sum(model.mid_block)

    # print("Calculating the sum of parameters in U-net Up Blocks")
    for idx, block in enumerate(model.up_blocks):
        sum_dict_up_blocks['ALL_Up_Block_'+str(idx)] = return_param_sum(block, False)

    return sum_dict_down_blocks, sum_dict_mid_block, sum_dict_up_blocks
    

def return_norm_param_sum(model, verbose=False):
    total_magnitude = 0
    
    for n,p in model.named_parameters():
        if p.requires_grad and "norm" in n:
            total_magnitude += p.sum().item()

    return round(total_magnitude, 3)

def return_blockwise_norm_param_sum(model, verbose=False):

    sum_dict_down_blocks = {}
    sum_dict_mid_block = {}
    sum_dict_up_blocks = {}

    # print("Calculating the sum of NORM parameters in U-net Down Blocks")
    for idx, block in enumerate(model.down_blocks):
        sum_dict_down_blocks['NORM_Down_Block_'+str(idx)] = return_norm_param_sum(block, False)

    # print("Calculating the sum of NORM parameters in U-net Mid Block")
    sum_dict_mid_block['NORM_Mid_Block'] = return_norm_param_sum(model.mid_block)

    # print("Calculating the sum of NORM parameters in U-net Up Blocks")
    for idx, block in enumerate(model.up_blocks):
        sum_dict_up_blocks['NORM_Up_Block_'+str(idx)] = return_norm_param_sum(block, False)

    return sum_dict_down_blocks, sum_dict_mid_block, sum_dict_up_blocks

def return_bias_param_sum(model, verbose=False):
    total_magnitude = 0
    
    for n,p in model.named_parameters():
        if p.requires_grad and "bias" in n:
            total_magnitude += p.sum().item()

    return round(total_magnitude, 3)

def return_blockwise_bias_param_sum(model, verbose=False):

    sum_dict_down_blocks = {}
    sum_dict_mid_block = {}
    sum_dict_up_blocks = {}

    # print("Calculating the sum of BIAS parameters in U-net Down Blocks")
    for idx, block in enumerate(model.down_blocks):
        sum_dict_down_blocks['BIAS_Down_Block_'+str(idx)] = return_bias_param_sum(block, False)

    # print("Calculating the sum of BIAS parameters in U-net Mid Block")
    sum_dict_mid_block['BIAS_Mid_Block'] = return_bias_param_sum(model.mid_block)

    # print("Calculating the sum of BIAS parameters in U-net Up Blocks")
    for idx, block in enumerate(model.up_blocks):
        sum_dict_up_blocks['BIAS_Up_Block_'+str(idx)] = return_bias_param_sum(block, False)

    return sum_dict_down_blocks, sum_dict_mid_block, sum_dict_up_blocks

def return_attn_param_sum(model, verbose=False):
    total_magnitude = 0
    
    for n,p in model.named_parameters():
        if p.requires_grad and "attentions" in n:
            total_magnitude += p.sum().item()

    return round(total_magnitude, 3)

def return_blockwise_attn_param_sum(model, verbose=False):

    sum_dict_down_blocks = {}
    sum_dict_mid_block = {}
    sum_dict_up_blocks = {}

    # print("Calculating the sum of ATTENTION parameters in U-net Down Blocks")
    for idx, block in enumerate(model.down_blocks):
        sum_dict_down_blocks['ATTN_Down_Block_'+str(idx)] = return_attn_param_sum(block, False)

    # print("Calculating the sum of ATTENTION parameters in U-net Mid Block")
    sum_dict_mid_block['ATTN_Mid_Block'] = return_attn_param_sum(model.mid_block)

    # print("Calculating the sum of ATTENTION parameters in U-net Up Blocks")
    for idx, block in enumerate(model.up_blocks):
        sum_dict_up_blocks['ATTN_Up_Block_'+str(idx)] = return_attn_param_sum(block, False)

    return sum_dict_down_blocks, sum_dict_mid_block, sum_dict_up_blocks

def check_params(model):
    print("The following parameters are trainable:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)


def check_tunable_params(model, verbose=True):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            if verbose:
                print(name)
            trainable_params += param.numel()

    # return  f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.5f} + \n"
    print("\n")
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.5f}"
    )

    return round(100 * trainable_params / all_param, 5)


# class conv_tsa(nn.Module):
#     def __init__(self, orig_conv, ad_type):
#         super().__init__()
#         # the original conv layer
#         self.conv = copy.deepcopy(orig_conv)
#         self.conv.weight.requires_grad = False
#         planes, in_planes, _, _ = self.conv.weight.size()
#         stride, _ = self.conv.stride
#         self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
#         self.alpha.requires_grad = True
#         self.ad_type = ad_type

#     def forward(self, x):
#         y = self.conv(x)
#         if self.ad_type == "residual":
#             if self.alpha.size(0) > 1:
#                 # residual adaptation in matrix form
#                 y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
#             else:
#                 # residual adaptation in channel-wise (vector)
#                 y = y + x * self.alpha
#         elif self.ad_type == "serial":
#             if self.alpha.size(0) > 1:
#                 # serial adaptation in matrix form
#                 y = F.conv2d(y, self.alpha) + self.alpha_bias
#             else:
#                 # serial adaptation in channel-wise (vector)
#                 y = y * self.alpha + self.alpha_bias
#         return y


# class pa(nn.Module):
#     """
#     pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
#     (https://arxiv.org/pdf/2103.13841.pdf)
#     """

#     def __init__(self, feat_dim):
#         super(pa, self).__init__()
#         # define pre-classifier alignment mapping
#         self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
#         self.weight.requires_grad = True

#     def forward(self, x):
#         if len(list(x.size())) == 2:
#             x = x.unsqueeze(-1).unsqueeze(-1)
#         x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
#         return x


# def apply_tsa_to_unet(unet, apply_tsa_downsampler=False, apply_tsa_upsampler=False):

#     print("Applying TSA to UNet")

#     # Attaching TSA in Down blocks
#     print("Iterating in Down Blocks")
#     i = 0
#     for block in unet.down_blocks:
#         i += 1
#         print("Block ", str(i))
#         resnets = block.resnets

#         if apply_tsa_downsampler:
#             # Down present only in Blocks 1,2,3
#             if i < 4:
#                 downsampler = block.downsamplers[0]
#                 setattr(downsampler, "conv", conv_tsa(downsampler.conv, "residual"))

#         for resnet in resnets:
#             setattr(resnet, "conv1", conv_tsa(resnet.conv1, "residual"))
#             setattr(resnet, "conv2", conv_tsa(resnet.conv2, "residual"))

#     # Attaching TSA in Mid blocks
#     for resnet in unet.mid_block.resnets:
#         setattr(resnet, "conv1", conv_tsa(resnet.conv1, "residual"))
#         setattr(resnet, "conv2", conv_tsa(resnet.conv2, "residual"))

#     print("\n")
#     print("Iterating in Up Blocks")
#     i = 0
#     for block in unet.up_blocks:
#         i += 1
#         print("Block ", str(i))
#         resnets = block.resnets

#         if apply_tsa_upsampler:
#             # Upsampler present only in Blocks 1,2,3
#             if i < 4:
#                 upsampler = block.upsamplers[0]
#                 setattr(upsampler, "conv", conv_tsa(upsampler.conv, "residual"))

#         for resnet in resnets:
#             setattr(resnet, "conv1", conv_tsa(resnet.conv1, "residual"))
#             setattr(resnet, "conv2", conv_tsa(resnet.conv2, "residual"))

#     return unet


# def apply_tsa_to_vae(vae, apply_tsa_downsampler=False, apply_tsa_upsampler=False):

#     print("Applying TSA to VAE")

#     # Attaching TSA in Encoder
#     i = 0
#     for block in vae.encoder.down_blocks:
#         i += 1
#         print("Block ", str(i))
#         resnets = block.resnets

#         if apply_tsa_downsampler:
#             # Downsampler present only in Block 1,2,3
#             if i < 4:
#                 downsampler = block.downsamplers[0]
#                 setattr(downsampler, "conv", conv_tsa(downsampler.conv, "residual"))

#         # Iterating in the resnet blocks of the encoder
#         for resnet in resnets:
#             setattr(resnet, "conv1", conv_tsa(resnet.conv1, "residual"))
#             setattr(resnet, "conv2", conv_tsa(resnet.conv2, "residual"))

#     for resnet in vae.encoder.mid_block.resnets:

#         setattr(resnet, "conv1", conv_tsa(resnet.conv1, "residual"))
#         setattr(resnet, "conv2", conv_tsa(resnet.conv2, "residual"))

#     # Attaching TSA in Decoder
#     i = 0
#     for block in vae.decoder.up_blocks:
#         i += 1
#         print("Block ", str(i))
#         resnets = block.resnets

#         if apply_tsa_upsampler:
#             # Upsampler present only in Block 1,2,3
#             if i < 4:
#                 upsampler = block.upsamplers[0]
#                 setattr(upsampler, "conv", conv_tsa(upsampler.conv, "residual"))

#         # Iterating in the resnet blocks of the decoder
#         for resnet in resnets:
#             setattr(resnet, "conv1", conv_tsa(resnet.conv1, "residual"))
#             setattr(resnet, "conv2", conv_tsa(resnet.conv2, "residual"))

#     for resnet in vae.decoder.mid_block.resnets:

#         setattr(resnet, "conv1", conv_tsa(resnet.conv1, "residual"))
#         setattr(resnet, "conv2", conv_tsa(resnet.conv2, "residual"))

#     return vae


def get_lora_config(text_lora_rank=16):
    config = {
        "peft_config": {
            "peft_type": "LORA",
            "task_type": None,
            "inference_mode": True,
            "r": 16,
            "target_modules": ["to_q", "to_v", "query", "value"],
            "lora_alpha": 27,
            "lora_dropout": 0.0,
            "merge_weights": False,
            "fan_in_fan_out": False,
            "enable_lora": None,
            "bias": "none",
        },
        "text_encoder_peft_config": {
            "peft_type": "LORA",
            "task_type": None,
            "inference_mode": True,
            "r": text_lora_rank,
            "target_modules": ["q_proj", "v_proj"],
            "lora_alpha": 17,
            "lora_dropout": 0.0,
            "merge_weights": False,
            "fan_in_fan_out": False,
            "enable_lora": None,
            "bias": "none",
        },
        "vae_peft_config": {
            "peft_type": "LORA",
            "task_type": None,
            "inference_mode": True,
            "r": 16,
            "target_modules": ["query", "value"],
            "lora_alpha": 17,
            "lora_dropout": 0.0,
            "merge_weights": False,
            "fan_in_fan_out": False,
            "enable_lora": None,
            "bias": "none",
        },
    }

    return config


# def apply_lora_to_unet(unet, lora_config=None):

#     """
#     Usage:
#             pipe.unet = apply_lora_to_unet(pipe.unet)

#     Number of trainable parameters should be around 0.18%
#     """

#     print("Applying LORA to UNet")

#     if lora_config is None:
#         lora_config = get_lora_config()

#     unet_config = LoraConfig(**lora_config["peft_config"])

#     print("Tunable parameters before applying LORA: ")
#     check_tunable_params(unet, verbose=False)

#     unet = LoraModel(unet_config, unet)

#     print("Tunable parameters after applying LORA: ")
#     check_tunable_params(unet, verbose=False)

#     return unet


def apply_lora_to_unetv2(unet, image_lora_rank, dtype):
    # Set correct lora layers
    # Snippet obtained from https://github.com/huggingface/diffusers/blob/7e6886f5e93ca9bb1e6d4beece46fe1e43b819c2/examples/text_to_image/train_text_to_image_lora.py#L446

    unet.requires_grad_(False)

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=image_lora_rank, dtype=dtype
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    return unet, lora_layers


def apply_svdiff_to_unet(args, **kwargs):

    # Adapted from https://github.com/mkshing/svdiff-pytorch/blob/a78f69e14410c1963318806050a566d262eca9f8/train_svdiff.py#L717
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    unet = load_unet_for_svdiff(pretrained_model_name_or_path, subfolder="unet", cache_dir=kwargs["cache_dir"])

    unet.requires_grad_(False)
    optim_params = []
    optim_params_1d = []
    for n, p in unet.named_parameters():
        if "delta" in n:
            p.requires_grad = True
            if "norm" in n:
                optim_params_1d.append(p)
            else:
                optim_params.append(p)

    return unet, optim_params, optim_params_1d


def apply_svdiff_to_text_encoder(args):

    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    text_encoder = load_text_encoder_for_svdiff(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )

    text_encoder.requires_grad_(False)
    optim_params = []
    optim_params_1d = []

    for n, p in text_encoder.named_parameters():
        if "delta" in n:
            p.requires_grad = True
            if "norm" in n:
                optim_params_1d.append(p)
            else:
                optim_params.append(p)

    return text_encoder, optim_params, optim_params_1d


def apply_lora_to_text_encoder(text_encoder, text_lora_rank=16, lora_config=None):

    """
    Usage:
            pipe.text_encoder = apply_lora_to_text_encoder(pipe.text_encoder)

    Number of trainable parameters should be around 0.47%
    """

    print("Applying LORA to Text Encoder")

    if lora_config is None:
        lora_config = get_lora_config(text_lora_rank=text_lora_rank)

    text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])

    print("Tunable parameters before applying LORA: ")
    check_tunable_params(text_encoder, verbose=False)

    text_encoder = LoraModel(text_encoder_config, text_encoder)

    print("Tunable parameters after applying LORA to text encoder: ")
    check_tunable_params(text_encoder, False)

    return text_encoder


# def apply_lora_to_vae(vae, lora_config=None):
#     """
#     Usage:
#             pipe.vae = apply_lora_to_vae(pipe.vae)

#     Number of trainable parameters should be around
#     """

#     print("Applying LORA to VAE")

#     if lora_config is None:
#         lora_config = get_lora_config()

#     vae_config = LoraConfig(**lora_config["vae_peft_config"])

#     print("Tunable parameters before applying LORA: ")
#     check_tunable_params(vae, verbose=False)

#     vae = LoraModel(vae_config, vae)

#     print("Tunable parameters after applying LORA")
#     check_tunable_params(vae, False)

#     return vae


def enable_bias_update(model):
    print("Enabling Bias layers")
    for m in model.modules():
        for name, param in m.named_parameters():
            if name == "bias":
                param.requires_grad = True

def mark_only_biases_as_trainable(model: nn.Module, is_bitfit=False):
    
    trainable_names = ["bias","norm","gamma","y_embed"]

    for par_name, par_tensor in model.named_parameters():
        par_tensor.requires_grad = any([kw in par_name for kw in trainable_names])

    return model


def enable_norm_update(model):
    print("Enabling Normalization layers")
    for m in model.modules():
        for name, param in m.named_parameters():
            if "norm" in name:
                param.requires_grad = True


def enable_attention_update(model):
    print("Enabling Attention layers")
    for m in model.modules():
        for name, param in m.named_parameters():
            if "attentions" in name:
                param.requires_grad = True

def disable_attention_update(model):
    print("Disabling Attention layers")
    for m in model.modules():
        for name, param in m.named_parameters():
            if "attentions" in name:
                param.requires_grad = False

def enable_blockwise_attention_update(unet, block_idx=None):

    if(block_idx is not None):
        assert block_idx <= 3, "Block index should be between 0 and 3"

    if(block_idx is None):
        # Update all blocks
        enable_attention_update(unet)
    else:
        # Update only the specified block
        enable_attention_update(unet[block_idx])



def enable_attention_update_text_encoder(text_encoder):
    print("Enabling Attention layers")
    for n, p in text_encoder.text_model.named_parameters():
        if "self_attn" in n:
            p.requires_grad = True

def get_adapted_unet(unet, method, args, **kwargs):
    if method == "attention":
        disable_grad(unet)
        enable_attention_update(unet)
        verbose = True
    elif method == "attention_down_blocks":
        disable_grad(unet)
        enable_blockwise_attention_update(unet.down_blocks, block_idx=kwargs["unet_block_idx"])
        verbose = True
    elif method == "attention_up_blocks":
        disable_grad(unet)
        enable_blockwise_attention_update(unet.up_blocks, block_idx=kwargs["unet_block_idx"])
        verbose = True
    elif method == "norm":
        disable_grad(unet)
        enable_norm_update(unet)
        verbose = True
    elif method == "bias":
        disable_grad(unet)
        enable_bias_update(unet)
        verbose = True
    elif method == "norm_bias":
        disable_grad(unet)
        enable_norm_update(unet)
        enable_bias_update(unet)
        verbose = True
    elif method == "norm_bias_attention":
        disable_grad(unet)
        enable_norm_update(unet)
        enable_bias_update(unet)
        enable_attention_update(unet)
        verbose = True
    elif method == "lora":
        unet = apply_lora_to_unet(unet)
        verbose = True
    elif method == "lorav2":
        unet, lora_layers = apply_lora_to_unetv2(unet, image_lora_rank=kwargs["image_lora_rank"], dtype=kwargs["dtype"])
        return unet, lora_layers
    elif method == "svdiff":
        unet, optim_params, optim_params_1d = apply_svdiff_to_unet(args, kwargs["cache_dir"])
        return unet, optim_params, optim_params_1d
    elif method == 'difffit':
        print("Loading model for DIFFFIT")
        unet = load_unet_for_difffit(
                pretrained_model_name_or_path=kwargs["pretrained_model_name_or_path"],
                efficient_weights_ckpt=None,
                hf_hub_kwargs=None,
                is_bitfit=False, 
                cache_dir=kwargs["cache_dir"]
                )
        unet = mark_only_biases_as_trainable(unet, is_bitfit=False)

    elif method == 'ssf':

        unet = load_unet_for_ssf(
                pretrained_model_name_or_path=kwargs["pretrained_model_name_or_path"],
                efficient_weights_ckpt=None,
                hf_hub_kwargs=None,
                is_bitfit=False, 
                )
        unet = mark_only_ssf_as_trainable(unet)

    elif method == 'oft':
        config_unet = OFTConfig(
            r=args.image_oft_rank,
            target_modules=[
                "proj_in",
                "proj_out",
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ],
            module_dropout=0.0,
            init_weights=True,
        )

        unet = OFTModel(unet, config_unet, "default")

    elif method == "loha":
        # https://arxiv.org/abs/2108.06098

        config_unet = LoHaConfig(
            r=args.image_loha_rank,
            alpha=32,
            target_modules=[
                "proj_in",
                "proj_out",
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ],
            rank_dropout=0.0,
            module_dropout=0.0,
            init_weights=True,
            use_effective_conv2d=True,
        )

        unet = LoHaModel(unet, config_unet, "default")

    elif method == 'lokr':
        # https://arxiv.org/abs/2108.06098
        # https://arxiv.org/abs/2309.14859

        config_unet = LoKrConfig(
            r=args.image_lokr_rank,
            alpha=32,
            target_modules=[
                "proj_in",
                "proj_out",
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ],
            rank_dropout=0.0,
            module_dropout=0.0,
            init_weights=True,
            use_effective_conv2d=True,
        )

        unet = LoKrModel(unet, config_unet, "default")

    elif method == "full":
        verbose = False
        pass
    elif method == "full_without_attention":
        verbose = False
        disable_attention_update(unet)
    elif method == "freeze":
        verbose = False
        disable_grad(unet)
    else:
        raise ValueError("Method not recognized")

    # check_tunable_params(unet, verbose)

    return unet


def get_adapted_text_encoder(text_encoder, method, args, **kwargs):

    # Training only attention was introduced in https://arxiv.org/abs/2203.09795
    if method == "attention":
        disable_grad(text_encoder)
        enable_attention_update_text_encoder(
            text_encoder
        )  # Around 28348416 params (23.036%) trainable
        verbose = True

    elif method == "norm":
        disable_grad(text_encoder)
        enable_norm_update(text_encoder)
        verbose = True

    # Training only bias terms in transformers was introduced in BitFit: https://arxiv.org/abs/2106.10199
    elif method == "bias":
        disable_grad(text_encoder)
        enable_bias_update(text_encoder)
        verbose = True

    elif method == "norm_bias":
        disable_grad(text_encoder)
        enable_norm_update(text_encoder)
        enable_bias_update(text_encoder)
        verbose = True

    elif method == "norm_bias_attention":
        disable_grad(text_encoder)
        enable_norm_update(text_encoder)
        enable_bias_update(text_encoder)
        enable_attention_update(text_encoder)
        verbose = True

    # LORA: https://arxiv.org/abs/2106.09685
    elif method == "lora":
        text_encoder = apply_lora_to_text_encoder(text_encoder, text_lora_rank=kwargs["text_lora_rank"])
        verbose = True

    elif method == "svdiff":
        text_encoder, optim_params, optim_params_1d = apply_svdiff_to_text_encoder(args)
        return text_encoder, optim_params, optim_params_1d

    elif method == "difffit":
        text_encoder = load_text_encoder_for_difffit(
                pretrained_model_name_or_path=kwargs["pretrained_model_name_or_path"],
                efficient_weights_ckpt=None,
                hf_hub_kwargs=None,
                is_bitfit=False, 
                )
        text_encoder = mark_only_biases_as_trainable(text_encoder, is_bitfit=False)

    elif method == "full":
        verbose = False
        pass

    elif method == "freeze":
        verbose = False
        disable_grad(text_encoder)

    else:
        raise ValueError("Method not recognized")

    return text_encoder


def get_adapted_vae(vae, method):

    if method == "tsa":
        # disable_grad(vae)
        # vae = apply_tsa_to_vae(
        #     vae, apply_tsa_downsampler=False, apply_tsa_upsampler=False
        # )  # 8.80% trainable
        # # vae = apply_tsa_to_vae(vae, apply_tsa_downsampler=True, apply_tsa_upsampler=True)  # 9.72% trainable
        # verbose = True
        raise NotImplementedError("TSA not implemented")

    elif method == "tsa2":
        # disable_grad(vae)
        # vae = apply_tsa_to_vae(
        #     vae, apply_tsa_downsampler=True, apply_tsa_upsampler=True
        # )
        # verbose = True
        raise NotImplementedError("TSA not implemented")

    elif method == "attention":
        # disable_grad(vae)
        # enable_attention_update(vae)  # Around 2103296 params (2.51%) trainable
        # verbose = True
        raise NotImplementedError("Attention Tuning not implemented for VAE")

    elif method == "lora":
        # vae = apply_lora_to_vae(
        #     vae, lora_config=None
        # )  # Around 65536 params (0.07%) trainable
        # verbose = True
        raise NotImplementedError("LoRA not implemented for VAE")

    elif method == "full":
        verbose = False
        pass

    elif method == "freeze":
        verbose = False
        disable_grad(vae)

    # check_tunable_params(vae, verbose)

    return vae


############## DiffFit ################

def mark_only_biases_as_trainable(model: nn.Module, is_bitfit=False):
    
    trainable_names = ["bias","norm","gamma","y_embed"]

    for par_name, par_tensor in model.named_parameters():
        par_tensor.requires_grad = any([kw in par_name for kw in trainable_names])

    return model

############## SSF ################

    
def mark_only_ssf_as_trainable(model):
    print("Enabling shift and scale parameters")
    for m in model.modules():
        for name, param in m.named_parameters():
            #print(name)
            # if "shift" in name or "scale" in name:
            #     param.requires_grad = True
            if "ssf_" in name:
                param.requires_grad = True

    return model



def get_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        par_name: par_tensor
        for par_name, par_tensor in model.named_parameters()
        if par_tensor.requires_grad
    }


def load_config_for_difffit(model_path, **kwargs):
    if os.path.exists(model_path):
        if "config.json" not in model_path:
            model_path = os.path.join(model_path, "config.json")
    else:
        model_path = huggingface_hub.hf_hub_download(model_path, filename="config.json", **kwargs)
    with open(model_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def load_unet_for_ssf(
    pretrained_model_name_or_path,
        efficient_weights_ckpt=None, 
        hf_hub_kwargs=None, 
        is_bitfit=False, 
        subfolder='unet',
        **kwargs
    ):
    # load pre-trained weights
    param_device = "cpu"
    torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None

    if not is_bitfit:
        config = UNet2DConditionModel.load_config(pretrained_model_name_or_path, subfolder=subfolder)
        original_model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        state_dict = original_model.state_dict()
        with accelerate.init_empty_weights():
            #model = UNet2DConditionModelForDiffFit.from_config(config)
            model = UNet2DConditionModelForSSF.from_config(config)
            # Set correct lora layers
            ssf_attn_procs = {}
            for name in model.attn_processors.keys():
                if name.startswith("mid_block"):
                    hidden_size = model.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(model.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = model.config.block_out_channels[block_id]
                
                ssf_attn_procs[name] = SSFAttnProcessor(hidden_size=hidden_size)

            model.set_attn_processor(ssf_attn_procs)

        scale_factor_weights = {n: torch.ones(p.shape) for n, p in model.named_parameters() if "ssf_" in n}
        state_dict.update(scale_factor_weights)
        # move the params from meta device to cpu
        missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_keys) > 0:
            raise ValueError(
                f"Cannot load {model.__class__.__name__} from {pretrained_model_name_or_path} because the following keys are"
                f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize"
                " those weights or else make sure your checkpoint file is correct."
            )
        for param_name, param in state_dict.items():
            accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
            if accepts_dtype:
                set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=torch_dtype)
            else:
                set_module_tensor_to_device(model, param_name, param_device, value=param)
    
    else:
        original_model = None
        model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
    
    if efficient_weights_ckpt:
        if os.path.isdir(efficient_weights_ckpt):
            efficient_weights_ckpt = os.path.join(efficient_weights_ckpt, "efficient_weights.safetensors")
        elif not os.path.exists(efficient_weights_ckpt):
            # download from hub
            hf_hub_kwargs = {} if hf_hub_kwargs is None else hf_hub_kwargs
            efficient_weights_ckpt = huggingface_hub.hf_hub_download(efficient_weights_ckpt, filename="efficient_weights.safetensors", **hf_hub_kwargs)
        assert os.path.exists(efficient_weights_ckpt)

        with safe_open(efficient_weights_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                # spectral_shifts_weights[key] = f.get_tensor(key)
                accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                if accepts_dtype:
                    set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key), dtype=torch_dtype)
                else:
                    set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key))
        print(f"Resumed from {efficient_weights_ckpt}")
    if "torch_dtype"in kwargs:
        model = model.to(kwargs["torch_dtype"])
    model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    del original_model
    torch.cuda.empty_cache()
    return model



def load_unet_for_difffit(
        pretrained_model_name_or_path,
        efficient_weights_ckpt=None, 
        hf_hub_kwargs=None, 
        is_bitfit=False, 
        subfolder='unet',
        **kwargs
    ):
    """
    https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/models/modeling_utils.py#L541
    """
    # load pre-trained weights
    param_device = "cpu"
    torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None
    cache_dir = kwargs["cache_dir"] if "cache_dir" in kwargs else None
    if not is_bitfit:
        config = UNet2DConditionModel.load_config(pretrained_model_name_or_path, subfolder=subfolder)
        original_model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, cache_dir=cache_dir)
        state_dict = original_model.state_dict()
        with accelerate.init_empty_weights():
            model = UNet2DConditionModelForDiffFit.from_config(config)
            # Set correct lora layers
            difffit_attn_procs = {}
            for name in model.attn_processors.keys():
                if name.startswith("mid_block"):
                    hidden_size = model.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(model.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = model.config.block_out_channels[block_id]

                difffit_attn_procs[name] = DiffFitAttnProcessor(hidden_size=hidden_size)

            model.set_attn_processor(difffit_attn_procs)
            
        scale_factor_weights = {n: torch.ones(p.shape) for n, p in model.named_parameters() if "gamma_" in n}
        state_dict.update(scale_factor_weights)
        # move the params from meta device to cpu
        missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_keys) > 0:
            raise ValueError(
                f"Cannot load {model.__class__.__name__} from {pretrained_model_name_or_path} because the following keys are"
                f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize"
                " those weights or else make sure your checkpoint file is correct."
            )
        for param_name, param in state_dict.items():
            accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
            if accepts_dtype:
                set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=torch_dtype)
            else:
                set_module_tensor_to_device(model, param_name, param_device, value=param)
    else:
        original_model = None
        model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, cache_dir=cache_dir)

    if efficient_weights_ckpt:
        if os.path.isdir(efficient_weights_ckpt):
            efficient_weights_ckpt = os.path.join(efficient_weights_ckpt, "efficient_weights.safetensors")
        elif not os.path.exists(efficient_weights_ckpt):
            # download from hub
            hf_hub_kwargs = {} if hf_hub_kwargs is None else hf_hub_kwargs
            efficient_weights_ckpt = huggingface_hub.hf_hub_download(efficient_weights_ckpt, filename="efficient_weights.safetensors", **hf_hub_kwargs)
        assert os.path.exists(efficient_weights_ckpt)

        with safe_open(efficient_weights_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                # spectral_shifts_weights[key] = f.get_tensor(key)
                accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                if accepts_dtype:
                    set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key), dtype=torch_dtype)
                else:
                    set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key))
        print(f"Resumed from {efficient_weights_ckpt}")
    if "torch_dtype"in kwargs:
        model = model.to(kwargs["torch_dtype"])
    model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    del original_model
    torch.cuda.empty_cache()
    return model

# def load_unet_for_ssf(
#         pretrained_model_name_or_path,
#         efficient_weights_ckpt=None, 
#         hf_hub_kwargs=None, 
#         is_bitfit=False, 
#         subfolder='unet',
#         **kwargs
#     ):
    

    #TODO: Implement SSF for UNet here



def load_text_encoder_for_difffit(
        pretrained_model_name_or_path,
        efficient_weights_ckpt=None,
        hf_hub_kwargs=None,
        is_bitfit=False,
        subfolder='text_encoder',
        **kwargs
    ):
    """
    https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/models/modeling_utils.py#L541
    """
    # load pre-trained weights
    print("################################ " + pretrained_model_name_or_path)
    param_device = "cpu"
    torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None
    if not is_bitfit:
        config = CLIPTextConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        original_model = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        state_dict = original_model.state_dict()
        with accelerate.init_empty_weights():
            model = CLIPTextModelForDiffFit(config)
        scale_factor_weights = {n: torch.ones(p.shape) for n, p in model.named_parameters() if "gamma_" in n}
        state_dict.update(scale_factor_weights)
        # move the params from meta device to cpu
        missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
        # if len(missing_keys) > 0:
        #     raise ValueError(
        #         f"Cannot load {model.__class__.__name__} from {pretrained_model_name_or_path} because the following keys are"
        #         f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
        #         " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize"
        #         " those weights or else make sure your checkpoint file is correct."
        #     )

        for param_name, param in state_dict.items():
            accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
            if accepts_dtype:
                set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=torch_dtype)
            else:
                set_module_tensor_to_device(model, param_name, param_device, value=param)
    else:
        original_model = None
        model = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)

    if efficient_weights_ckpt:
        if os.path.isdir(efficient_weights_ckpt):
            efficient_weights_ckpt = os.path.join(efficient_weights_ckpt, "efficient_weights_te.safetensors")
        elif not os.path.exists(efficient_weights_ckpt):
            # download from hub
            hf_hub_kwargs = {} if hf_hub_kwargs is None else hf_hub_kwargs
            try:
                efficient_weights_ckpt = huggingface_hub.hf_hub_download(efficient_weights_ckpt, filename="efficient_weights_te.safetensors", **hf_hub_kwargs)
            except huggingface_hub.utils.EntryNotFoundError:
                efficient_weights_ckpt = None
        # load state dict only if `spectral_shifts_te.safetensors` exists
        if os.path.exists(efficient_weights_ckpt):
            with safe_open(efficient_weights_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # spectral_shifts_weights[key] = f.get_tensor(key)
                    accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                    if accepts_dtype:
                        set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key), dtype=torch_dtype)
                    else:
                        set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key))
            print(f"Resumed from {efficient_weights_ckpt}")
        
    if "torch_dtype"in kwargs:
        model = model.to(kwargs["torch_dtype"])
    # model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    del original_model
    torch.cuda.empty_cache()
    return model

############################################ NEW FUNCTIONS FOR THE HPO ############################################

##### 1. SV-DIFF

def return_new_optim_params(unet):
    optim_params_1d = []
    optim_params = []
    for n,p in unet.named_parameters():
        if("delta" in n and p.requires_grad == True):
            if "norm" in n:
                optim_params_1d.append(p)
            else:
                optim_params.append(p)

    return optim_params, optim_params_1d

    
def enable_disable_svdiff_with_mask(unet_with_svdiff, binary_mask):
    attention_pattern = [0,1]

    mid_block_mask = [binary_mask[6]]
    down_block_mask = binary_mask[0:6]
    up_block_mask = binary_mask[7:]
    
    # Down Blocks
    print("Down Blocks")
    for i in range(len(down_block_mask)):
        block_idx = i//2
        attention_idx = attention_pattern[i%2]
        mask_element = down_block_mask[i]
    
        print("Block: ", block_idx)
        print("Attention: ", attention_idx)
    
        if(mask_element == 0):
            disable_grad_svdiff(unet_with_svdiff.down_blocks[block_idx].attentions[attention_idx].transformer_blocks)
        else:
            continue # Elements are trainable by default
    
    # Mid Block
    print("Mid Blocks")
    for i in range(len(mid_block_mask)):
        mask_element = mid_block_mask[i]
    
        if(mask_element == 0):
            disable_grad_svdiff(unet_with_svdiff.mid_block.attentions[0].transformer_blocks)
    
    # Up Blocks
    print("Up Blocks")
    for i in range(len(up_block_mask)):
        mask_element = up_block_mask[i] 
        block_idx = (i//2)+1 # Indexing for up-blocks starts from 1
        attention_idx = attention_pattern[i%2]
        print("Block: ", block_idx)
        print("Attention: ", attention_idx)
    
        if(mask_element == 0):
            disable_grad_svdiff(unet_with_svdiff.up_blocks[block_idx].attentions[attention_idx].transformer_blocks)

    optim_params, optim_params_1d = return_new_optim_params(unet_with_svdiff)

    return unet_with_svdiff, optim_params, optim_params_1d

##### 2. DiffFit
def enable_disable_difffit_with_mask(unet, binary_mask, verbose=False):
    mid_block_mask = [binary_mask[6]]
    down_block_mask = binary_mask[0:6]
    up_block_mask = binary_mask[7:]

    # Sanity Check
    assert (len(mid_block_mask) + len(down_block_mask) + len(up_block_mask)) == len(binary_mask)

    ################# DOWN BLOCKS #################
    if(verbose):
        print("Masking down blocks")
    attention_pattern = [0,1]
    for i in range(len(down_block_mask)):
        block_idx = i//2
        attention_idx = attention_pattern[i%2]
        mask_element = down_block_mask[i]

        if(verbose):
            print("Block: ", block_idx)
            print("Attention: ", attention_idx)
    
        if(mask_element == 0):
            disable_grad_difffit(unet.down_blocks[block_idx].attentions[attention_idx].transformer_blocks)
        else:
            continue # Elements are trainable by default

    ################# MID BLOCK #################
    if(verbose):
        print("Masking Mid block")
        
    for i in range(len(mid_block_mask)):
        mask_element = mid_block_mask[i]
    
        if(mask_element == 0):
            disable_grad_difffit(unet.mid_block.attentions[0].transformer_blocks)

    ################# UP BLOCKS #################
    if(verbose):
        print("Masking Up blocks")

    attention_pattern = [0,1]
    for i in range(len(up_block_mask)):
        mask_element = up_block_mask[i] 
        block_idx = (i//2)+1 # Indexing for up-blocks starts from 1
        attention_idx = attention_pattern[i%2]
        if(verbose):
            print("Block: ", block_idx)
            print("Attention: ", attention_idx)
    
        if(mask_element == 0):
            disable_grad_difffit(unet.up_blocks[block_idx].attentions[attention_idx].transformer_blocks)

    return unet
    
def enable_disable_attention_with_mask(unet, binary_mask, verbose=False):
    mid_block_mask = [binary_mask[6]]
    down_block_mask = binary_mask[0:6]
    up_block_mask = binary_mask[7:]

    

    # Sanity Check
    assert (len(mid_block_mask) + len(down_block_mask) + len(up_block_mask)) == len(binary_mask)

    ################# DOWN BLOCKS #################
    if(verbose):
        print("Masking down blocks")
    attention_pattern = [0,1]
    for i in range(len(down_block_mask)):
        block_idx = i//2
        attention_idx = attention_pattern[i%2]
        mask_element = down_block_mask[i]

        if(verbose):
            print("Block: ", block_idx)
            print("Attention: ", attention_idx)
    
        if(mask_element == 0):
            disable_grad_attention(unet.down_blocks[block_idx].attentions[attention_idx].transformer_blocks)
        else:
            continue # Elements are trainable by default

    ################# MID BLOCK #################
    if(verbose):
        print("Masking Mid block")
        
    for i in range(len(mid_block_mask)):
        mask_element = mid_block_mask[i]
    
        if(mask_element == 0):
            disable_grad_attention(unet.mid_block.attentions[0].transformer_blocks)

    ################# UP BLOCKS #################
    if(verbose):
        print("Masking Up blocks")

    attention_pattern = [0,1,2]
    for i in range(len(up_block_mask)):
        mask_element = up_block_mask[i] 
        block_idx = (i//3)+1 # Indexing for up-blocks starts from 1
        attention_idx = attention_pattern[i%3]
        if(verbose):
            print("Block: ", block_idx)
            print("Attention: ", attention_idx)
    
        if(mask_element == 0):
            disable_grad_attention(unet.up_blocks[block_idx].attentions[attention_idx].transformer_blocks)

    return unet