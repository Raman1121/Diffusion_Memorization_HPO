from pathlib import Path
import pandas as pd
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import torch
import argparse
import pandas as pd
import os
import random
import warnings
from parse_args import parse_args
from safetensors.torch import load_file
import yaml

from svdiff.utils import (
    load_unet_for_svdiff,
    load_text_encoder_for_svdiff,
    SCHEDULER_MAPPING,
)

# from metrics.metrics import *

warnings.filterwarnings("ignore")

def load_adapted_unet(args, exp_path, pipe):
    sd_folder_path = args["pretrained_model_name_or_path"]

    if args["unet_pretraining_type"] == "freeze":
        pass
    
    elif(args["unet_pretraining_type"] == "svdiff" or args["unet_pretraining_type"] == "auto_svdiff"):
        print("SV-DIFF UNET")

        pipe.unet = load_unet_for_svdiff(
            sd_folder_path,
            spectral_shifts_ckpt=os.path.join(os.path.join(exp_path, 'unet'), "spectral_shifts.safetensors"),
            subfolder="unet",
        )
        for module in pipe.unet.modules():
            if hasattr(module, "perform_svd"):
                module.perform_svd()

    else:
        try:
            exp_path = os.path.join(exp_path, 'unet', 'diffusion_pytorch_model.safetensors')
            state_dict = load_file(exp_path)
            print(pipe.unet.load_state_dict(state_dict, strict=False))
        except:
            import pdb; pdb.set_trace()


def loadSDModel(args, exp_path, cuda_device):

    device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    sd_folder_path = args["pretrained_model_name_or_path"]

    pipe = StableDiffusionPipeline.from_pretrained(sd_folder_path, revision=args["mixed_precision"])

    load_adapted_unet(args, exp_path, pipe)

    pipe.to(device)
    pipe.to(torch.float16)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    return pipe

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # self.df = pd.read_excel(prompts_path)[["text", "path", "subject_id", "study"]]
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return dict(self.df.iloc[idx])

def generate_synthetic_dataset(args, df):
    
    args["save_images_path"] = os.path.join(args["exp_path"], "synthetic_images_{}".format(args["run_eval_on"]))
    if not os.path.isdir(args["save_images_path"]):
        os.makedirs(args["save_images_path"])

    sd_pipeline = loadSDModel(
        args, exp_path=args["output_dir"], cuda_device=args["cuda_device"]
    )

    dataset = TextDataset(
        df,
    )

    text_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args["train_batch_size"], num_workers=2
    )

    df_out = pd.DataFrame(
        columns=["subject_id", "study", "gt_image_path", "path", "text"]
    )

    for epoch in range(args["images_per_prompt"]):
        for batch in text_loader:
            result = sd_pipeline(
                prompt=batch["text"],
                height=args["resolution"],
                width=args["resolution"],
                guidance_scale=4,
                num_inference_steps=50,
            )
            batch["gt_image_path"] = []
            for i, img in enumerate(result.images):
                root_name = os.path.join(
                    args["save_images_path"],
                    batch["path"][i].split("/")[-1].split(".")[0],
                )
                save_image_path = root_name + ".jpg"
                batch["gt_image_path"].append(save_image_path)
                img.save(save_image_path)
            df_out = pd.concat([df_out, pd.DataFrame.from_dict(batch)])
    df_out.reset_index(drop=True)
    df_out.to_csv(
        os.path.join(args["save_images_path"], 'samples_info.csv')
    )

def generate_and_eval(args):

    with open("data_config.yaml") as file:
        yaml_data = yaml.safe_load(file)

    if(args["run_eval_on"] == 'train'):
        print("Generating images using prompts from TRAINING DATA")
        args["prompts_path"] = yaml_data["train_csv"] 
    elif(args["run_eval_on"] == 'test'):
        print("Generating images using prompts from TEST DATA")
        args["prompts_path"] = yaml_data["test_csv"]
    else:
        raise ValueError("Invalid value for run_eval_on. Select from 'train' or 'test' only.")

    # TODO: Add the logic of running generation and evaluation across different seeds here

    # SEEDS = [42, 1234, 5678, 1111]
    SEEDS = [42]

    for seed in SEEDS:
        random.seed(seed)

        # Subset the dataframe (1000 samples) randomly according to the seed
        df = pd.read_excel(args["prompts_path"])
        df = df.sample(n=args["num_images_to_generate"], random_state=seed)
        df = df.reset_index(drop=True)

        # Generate synthetic images and save them
        generate_synthetic_dataset(args, df)


if __name__ == "__main__":

    config = parse_args()
    project_root_path = Path(os.getcwd())

    config.output_dir = os.path.join(
            config.output_dir, 
            config.unet_pretraining_type
        )
    config.cuda_device = 0

    config.exp_path = config.output_dir
    print(config.exp_path)

    generate_and_eval(vars(config))
