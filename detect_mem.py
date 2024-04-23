import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd

import torch

from optim_utils import *
from io_utils import *

from local_sd_pipeline import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel

import yaml
# from parse_args import get_config
from get_dataset_mimic_cxr import MimicCXRDataset

def main(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open("unet_peft.yaml") as file:
        yaml_data_peft = yaml.safe_load(file)

    args.unet_id = yaml_data_peft[args.peft_method]

    print("PEFT Method: ", args.peft_method)
    print("Loading UNet model: ", args.unet_id)

    if args.unet_id is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_id, torch_dtype=torch.float16
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # dataset
    set_random_seed(args.gen_seed)
    # dataset, prompt_key = get_dataset(args.dataset, pipe)

    # MIMIC DATASET HERE
    # Import CSV path from the YAML file
    with open("data_config.yaml") as file:
        yaml_data = yaml.safe_load(file)

    args.train_data_path = yaml_data["train_csv"]
    args.val_data_path = yaml_data["val_csv"]
    args.test_data_path = yaml_data["test_csv"]
    args.counts_data_path = yaml_data["counts_csv"]

    args.images_path_train = Path(yaml_data["images_path_train"])
    args.images_path_val = Path(yaml_data["images_path_val"])

    if(args.run_on_frequent_samples or args.run_on_rare_samples):
        # df = pd.read_excel(args.train_data_path)
        print("Reading Counts Dataframe")
        df = pd.read_csv(args.counts_data_path)
        ALL_COUNTS = df['Count'].to_list()
        ALL_UNIQUE_PROMPTS = list(set(df['Text'].tolist()))
    else:
        df = pd.read_excel(args.train_data_path)

    if(args.use_findings):
        ALL_UNIQUE_PROMPTS = list(set(df['findings'].tolist()))
    else:
        if(args.run_on_frequent_samples):
            if(args.by_percentile):
                percentile_index = int(len(ALL_COUNTS) * 0.05)  # Selecting prompts in the top 5 percentile by frequncy
                ALL_UNIQUE_PROMPTS = ALL_UNIQUE_PROMPTS[:percentile_index]
            else:
                ALL_UNIQUE_PROMPTS = list(set(df['Text'].tolist()))
                # Select top 50
                ALL_UNIQUE_PROMPTS = ALL_UNIQUE_PROMPTS[:50]
        elif(args.run_on_rare_samples):
            if(args.by_percentile):
                percentile_index = int(len(ALL_COUNTS) * 0.05) 
                ALL_UNIQUE_PROMPTS = ALL_UNIQUE_PROMPTS[-percentile_index:]
            else:
                ALL_UNIQUE_PROMPTS = list(set(df['Text'].tolist()))
                # Select bottom 50
                ALL_UNIQUE_PROMPTS = ALL_UNIQUE_PROMPTS[-50:]
        else:
            ALL_UNIQUE_PROMPTS = list(set(df['text'].tolist()))

    if(args.run_on_full_dataset):
        args.end = len(ALL_UNIQUE_PROMPTS)
    else:
        args.end = min(args.end, len(df))

    # generation
    print("generation")

    all_metrics = ["uncond_noise_norm", "text_noise_norm"]
    all_tracks = []

    for i in tqdm(range(args.start, args.end)):
        try:
            seed = i + args.gen_seed

            # prompt = dataset[i][prompt_key]
            # prompt = train_dataset[i]["text"]
            prompt = ALL_UNIQUE_PROMPTS[i]
            print("Prompt: ", prompt)

            ### generation
            set_random_seed(seed)
            outputs, track_stats = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                track_noise_norm=True,
            )

            uncond_noise_norm, text_noise_norm = (
                track_stats["uncond_noise_norm"],
                track_stats["text_noise_norm"],
            )

            curr_line = {}
            for metric_i in all_metrics:
                values = locals()[metric_i]
                curr_line[f"{metric_i}"] = values

            curr_line["prompt"] = prompt

            all_tracks.append(curr_line)
            print("\n")

        except:
            continue

    os.makedirs("det_outputs", exist_ok=True)
    # write_jsonlines(all_tracks, f"det_outputs/{args.run_name}.jsonl")

    if(args.run_on_frequent_samples):
        write_jsonlines(all_tracks, "det_outputs/{}_frequent.jsonl".format(args.run_name + "_" + args.peft_method))
    elif(args.run_on_rare_samples):
        write_jsonlines(all_tracks, "det_outputs/{}_rare.jsonl".format(args.run_name + "_" + args.peft_method))
    else:
        write_jsonlines(all_tracks, "det_outputs/{}.jsonl".format(args.run_name + "_" + args.peft_method))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--run_name", default="test")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=500, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_id", default=None)
    parser.add_argument("--peft_method", type=str, default='full')
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--num_images_per_prompt", default=4, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--run_on_full_dataset", action="store_true")
    parser.add_argument("--use_findings", action="store_true")
    parser.add_argument("--run_on_frequent_samples", action="store_true")
    parser.add_argument("--run_on_rare_samples", action="store_true")
    parser.add_argument("--by_percentile", action="store_true")

    args = parser.parse_args()

    main(args)
