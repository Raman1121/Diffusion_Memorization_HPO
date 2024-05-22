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
import glob
import shutil
from metrics_utils import *
from adaptors import check_tunable_params
from torchmetrics.image.fid import FrechetInceptionDistance
import torchxrayvision as xrv
import torchvision

from svdiff.utils import (
    load_unet_for_svdiff,
    load_text_encoder_for_svdiff,
    SCHEDULER_MAPPING,
)

# from metrics.metrics import *

warnings.filterwarnings("ignore")

#################### IMAGE GENERATION FUNCTIONS ####################


def load_adapted_unet(args, exp_path, pipe):
    sd_folder_path = args["pretrained_model_name_or_path"]

    if args["unet_pretraining_type"] == "freeze":
        pass

    elif (
        args["unet_pretraining_type"] == "svdiff"
        or args["unet_pretraining_type"] == "auto_svdiff"
    ):
        print("SV-DIFF UNET")

        pipe.unet = load_unet_for_svdiff(
            sd_folder_path,
            spectral_shifts_ckpt=os.path.join(
                os.path.join(exp_path, "unet"), "spectral_shifts.safetensors"
            ),
            subfolder="unet",
        )
        for module in pipe.unet.modules():
            if hasattr(module, "perform_svd"):
                module.perform_svd()

    else:
        try:
            exp_path = os.path.join(
                exp_path, "unet", "diffusion_pytorch_model.safetensors"
            )
            state_dict = load_file(exp_path)
            print(pipe.unet.load_state_dict(state_dict, strict=False))
        except:
            import pdb

            pdb.set_trace()


def loadSDModel(args, exp_path, cuda_device):

    # device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    # device = "cuda:1"
    sd_folder_path = args["pretrained_model_name_or_path"]

    pipe = StableDiffusionPipeline.from_pretrained(
        sd_folder_path, revision=args["mixed_precision"]
    )

    if args["unet_pretraining_type"] != "freeze":
        load_adapted_unet(args, exp_path, pipe)
    else:
        pass

    pipe.to("cuda")
    # pipe.to(torch.float16)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    tunable_params = check_tunable_params(pipe.unet, False)

    return pipe, tunable_params


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # self.df = pd.read_excel(prompts_path)[["text", "path", "subject_id", "study"]]
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return dict(self.df.iloc[idx])


def generate_synthetic_dataset(args, df, sd_pipeline):

    if not os.path.isdir(args["save_images_path"]):
        os.makedirs(args["save_images_path"])

    # sd_pipeline = loadSDModel(
    #     args, exp_path=args["output_dir"], cuda_device=args["cuda_device"]
    # )

    dataset = TextDataset(
        df,
    )

    text_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args["train_batch_size"], num_workers=16, drop_last=False
    )
    print("Batch Size: ", args["train_batch_size"])

    df_out = pd.DataFrame(
        columns=["subject_id", "study", "gt_image_path", "path", "text"]
    )

    for epoch in range(args["images_per_prompt"]):
        for batch in text_loader:
            with torch.autocast("cuda"):
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
    df_out.to_csv(os.path.join(args["save_images_path"], "samples_info.csv"))


def generate_and_eval(args):

    with open("data_config.yaml") as file:
        yaml_data = yaml.safe_load(file)

    if args["run_eval_on"] == "train":
        print("Generating images using prompts from TRAINING DATA")
        args["prompts_path"] = yaml_data[args["dataset"]]["train_csv"]

    elif args["run_eval_on"] == "test":
        print("Generating images using prompts from TEST DATA")

        if args["dataset"] == "MIMIC":
            args["prompts_path"] = yaml_data[args["dataset"]]["test_csv"]
        elif args["dataset"] == "imagenette":
            args["prompts_path"] = yaml_data[args["dataset"]]["val_csv"]

        else:
            raise ValueError(
                "Invalid value for dataset. Select from 'MIMIC' or 'imagenette' only."
            )
    else:
        raise ValueError(
            "Invalid value for run_eval_on. Select from 'train' or 'test' only."
        )

    sd_pipeline, tunable_params = loadSDModel(
        args, exp_path=args["output_dir"], cuda_device=args["cuda_device"]
    )

    # TODO: Add the logic of running generation and evaluation across different seeds here

    GLOBAL_FID = []
    GLOBAL_MIFID = []
    # SEEDS = [42, 1234, 5678, 1111]
    SEEDS = [42]

    for seed in SEEDS:
        random.seed(seed)

        # Subset the dataframe (1000 samples) randomly according to the seed

        if args["dataset"] == "MIMIC":
            df = pd.read_excel(args["prompts_path"])
        elif args["dataset"] == "imagenette":
            df = pd.read_csv(args["prompts_path"])

        df["path"] = df["path"].apply(
            lambda x: os.path.join(yaml_data[args["dataset"]]["images_path_train"], x)
        )
        # df = df.sample(n=args["num_images_to_generate"], random_state=seed)
        # Sample first 1000 samples
        df = df.iloc[:1000]
        df = df.reset_index(drop=True)

        # STEP 1: Generate synthetic images and save them
        print("Generating Synthetic Images")
        generate_synthetic_dataset(args, df, sd_pipeline)

        # STEP 2: Calculate metrics

        # Preparing real image tensors
        real_image_paths = df["path"].tolist()
        print("Preparing Real Image Tensors")
        real_images = get_images_tensor_from_paths(real_image_paths)

        # Preparing Synthetic Image tensors
        synthetic_image_paths = glob.glob(
            os.path.join(args["save_images_path"], "*.jpg")
        )

        print("{} Synthetic Images found".format(len(synthetic_image_paths)))
        print("Preparing Synthetic Image Tensors")
        synthetic_images = get_images_tensor_from_paths(synthetic_image_paths)

        # Calculate the MIFID Score
        if args["run_eval_on"] == "train":
            # TODO: Compute MIFID HERE
            mifid_score = compute_mifid(real_images, synthetic_images, device="cuda:0")
            GLOBAL_MIFID.append(mifid_score)
            print("MIFID SCORE is {} for Seed {}".format(mifid_score, seed))

        # Calculate the FID Score
        elif args["run_eval_on"] == "test":
            fid_score = compute_fid(real_images, synthetic_images, device="cuda:0")
            GLOBAL_FID.append(fid_score)
            print("FID SCORE is {} for Seed {}".format(fid_score, seed))

        print("\n")
        print("\n")

        # Removing synthetic images directory
        # shutil.rmtree(args["save_images_path"])

    if (
        args["run_eval_on"] == "train"
    ):  # If running eval on training data, we need to calculate MIFID
        try:
            results_df = pd.read_csv("results_MIFID.csv")
        except:
            results_df = pd.DataFrame(
                columns=["FT Strategy", "Tunable Params", "MIFID", "Standard Dev"]
            )

        # Add the results to the results dataframe
        _row = [
            args["unet_pretraining_type"],
            tunable_params,
            np.mean(GLOBAL_MIFID),
            np.std(GLOBAL_MIFID),
        ]
        results_df.loc[len(results_df)] = _row

        if args["use_random_word_addition"]:
            results_df.to_csv(
                os.path.join(args["results_savedir"], "results_MIFID.csv"), index=False
            )
        else:
            results_df.to_csv(
                os.path.join(args["results_savedir"], "results_MIFID.csv"), index=False
            )

    elif args["run_eval_on"] == "test":
        try:
            results_df = pd.read_csv("results_FID.csv")
        except:
            results_df = pd.DataFrame(
                columns=["FT Strategy", "Tunable Params", "FID", "Standard Dev"]
            )

        # Add the results to the results dataframe
        _row = [
            args["unet_pretraining_type"],
            tunable_params,
            np.mean(GLOBAL_FID),
            np.std(GLOBAL_FID),
        ]
        results_df.loc[len(results_df)] = _row

        if args["use_random_word_addition"]:
            results_df.to_csv(
                os.path.join(args["results_savedir"], "results_FID.csv"), index=False
            )
        else:
            results_df.to_csv(
                os.path.join(args["results_savedir"], "results_FID.csv"), index=False
            )


if __name__ == "__main__":

    config = parse_args()
    project_root_path = Path(os.getcwd())

    if config.use_random_word_addition:
        config.unet_pretraining_type = config.unet_pretraining_type + "_RWA"
    if config.mitigation_threshold is not None:
        config.unet_pretraining_type = (
            config.unet_pretraining_type
            + "_Mitigation_{}".format(config.mitigation_threshold)
        )

    # if config.use_random_word_addition:
    #     config.output_dir = os.path.join(
    #         config.output_dir,
    #         config.unet_pretraining_type + "_RWA"
    #     )
    # if(config.mitigation_threshold is not None):
    #     config.output_dir = os.path.join(
    #         config.output_dir,
    #         config.unet_pretraining_type + "_RWA"
    #     )
    # else:
    #     config.output_dir = os.path.join(
    #         config.output_dir,
    #         config.unet_pretraining_type
    #     )

    config.output_dir = os.path.join(config.output_dir, config.unet_pretraining_type)

    config.results_savedir = os.path.join(config.output_dir, "results")
    os.makedirs(config.results_savedir, exist_ok=True)

    # config.cuda_device = 0
    config.cuda_device = torch.cuda.current_device()
    print("CUDA Device: ", config.cuda_device)
    # config.train_batch_size = 16   # Works when the pipeline dtype is fp16

    config.exp_path = config.output_dir
    config.save_images_path = os.path.join(
        config.exp_path, "synthetic_images_{}".format(config.run_eval_on)
    )
    print(config.exp_path)

    generate_and_eval(vars(config))
