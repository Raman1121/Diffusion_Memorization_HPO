import argparse
import json
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from torchvision import transforms
from extraction_attack import attack, similarity
from datetime import datetime
import os
import pandas as pd

from svdiff.utils import (
    load_unet_for_svdiff,
    load_text_encoder_for_svdiff,
    SCHEDULER_MAPPING,
)
from safetensors.torch import load_file

def load_adapted_unet(args, exp_path, pipe):
    args = vars(args)
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

def loadSDModel(args, exp_path):

    # device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    # device = "cuda:1"

    args = vars(args)
    sd_folder_path = args["pretrained_model_name_or_path"]

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision=args["mixed_precision"])

    if(args["unet_pretraining_type"] != "freeze"):
        load_adapted_unet(args, exp_path, pipe)
    else:
        pass

    # pipe.to('cuda')
    # pipe.to(torch.float16)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    tunable_params = check_tunable_params(pipe.unet, False)

    return pipe, tunable_params

def evaluate(args, pipe):
    
    # Read the training df here
    try:
        df = pd.read_csv(args.train_df) 
    except:
        df = pd.read_excel(args.train_df)

    df = df.iloc[:10]
    df = df.reset_index(drop=True)

    image_paths = df['path'].tolist()
    image_paths = [os.path.join(args.samples_root, path) for path in image_paths]
    captions = df['text'].tolist()

    transform = transforms.ToTensor()
    transform_crop = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop((args.resolution, args.resolution)),
        transforms.ToTensor(),
    ])

    memorized_images = []
    similar_images = []
    num_memorized = 0
    num_similar = 0  # only for the memorized images
    total = len(image_paths)
    actual_total = 0


    if args.eval_all:
        # load all of the images into memory
        loaded_images = []
        loaded_captions = []
        loaded_image_paths = []

        for idx, (image_path, caption) in enumerate(zip(image_paths, captions)):
            try:
                image = Image.open(image_path).convert("RGB")
                image = transform_crop(image).unsqueeze(0)
                loaded_images.append(image)
                loaded_captions.append(caption)
                loaded_image_paths.append(image_path)
            except Exception as e:
                print(f"Error processing image at index {idx}: {e}")
        
        for idx, (image, caption, image_path) in enumerate(zip(loaded_images, loaded_captions, loaded_image_paths)):
            if idx % 100 == 0:
                print(f"Processed {idx}/{total} images at {datetime.now()}")

            # load the image from the path
            try:
                # get the height and width of the image
                prompt = [caption] * args.num_images_per_prompt
                # generate images for the prompt

                if args.load_samples:
                    images = []
                    for i in range(args.num_images_per_prompt):
                        # img = Image.open(args.samples_root + '/' + image_path.split('/')[0] + f"/samples/img_{idx}_sample_{i}.jpg")
                        img = Image.open(image_path)
                        images.append(img)

                else:
                    images = pipe(prompt, num_inference_steps=50, height=args.resolution, width=args.resolution).images

                image = image.to("cuda")  # shape is [1, 3, width, height]
                # convert the generated images to a tensor
                images = torch.cat([transform(img).unsqueeze(0) for img in images], dim=0).to("cuda")
                attack_index = attack(images, min_clique_size=args.min_clique_size, threshold=args.threshold)

                if attack_index != -1:
                    num_memorized += 1
                    memorized_images.append((image_path, caption))
                    for test_image in loaded_images:
                        # check if the memorized image is similar to the current image
                        if similarity(test_image.to("cuda")[0], images[attack_index], threshold=args.threshold):
                            num_similar += 1
                            similar_images.append((image_path, caption))
                            break
                actual_total += 1
            except Exception as e:
                print(f"Error processing image at index {idx}: {e}")    

    else:
        for idx, (image_path, caption) in enumerate(zip(image_paths, captions)):
            if idx % 100 == 0:
                print(f"Processed {idx}/{total} images at {datetime.now()}")

            # load the image from the path
            try:
                image = Image.open(image_path).convert("RGB")
                # get the height and width of the image
                prompt = [caption] * args.num_images_per_prompt
                # generate images for the prompt
                if args.load_samples:
                    images = []
                    for i in range(args.num_images_per_prompt):
                        # img = Image.open(args.samples_root + '/' + image_path.split('/')[0] + f"/samples/img_{idx}_sample_{i}.jpg")
                        img = Image.open(image_path)
                        images.append(img)
                else:
                    images = pipe(prompt, num_inference_steps=50, height=args.resolution, width=args.resolution).images

                image = transform_crop(image).unsqueeze(0).to("cuda")  # shape is [1, 3, width, height]
                # convert the generated images to a tensor
                images = torch.cat([transform(img).unsqueeze(0) for img in images], dim=0).to("cuda")

                attack_index = attack(images, min_clique_size=args.min_clique_size, threshold=args.threshold)

                if attack_index != -1:
                    num_memorized += 1
                    memorized_images.append((image_path, caption))

                    if similarity(image[0], images[attack_index], threshold=args.threshold):
                        num_similar += 1
                        similar_images.append((image_path, caption))
                actual_total += 1

            except Exception as e:
                print(f"Error processing image at index {idx}: {e}")


    return memorized_images, similar_images, {"total": total, "memorized": num_memorized, "similar": num_similar, "actual_total": actual_total}  

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find memorized images in a dataset.')
    parser.add_argument("--data_file", type=str, default="imagenette2-320/imagenette_captions.json")
    parser.add_argument("--mem_file", type=str, default="")  # the file could e.g. be results/deduplication.json
    parser.add_argument("--num_images_per_prompt", type=int, default=50)
    parser.add_argument("--min_clique_size", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--print_stats", type=int, default=1)
    parser.add_argument("--outputs_name", type=str, default="")
    parser.add_argument("--load_samples", type=int, default=0)
    parser.add_argument("--eval_all", type=int, default=0)
    parser.add_argument("--model", type=str, default="sdv15")
    

    # Add the following arguments
    parser.add_argument("--train_df", type=str, default="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSVs/FINAL_TRAIN.xlsx")
    parser.add_argument("--samples_root", type=str, default="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0")
    parser.add_argument("--output_dir", type=str, default="", required=True)
    parser.add_argument("--unet_pretraining_type", type=str, default="full", required=True)

    args = parser.parse_args()

    # pipe = StableDiffusionPipeline.from_pretrained(args.output_dir, torch_dtype=torch.float32, safety_checker = None, requires_safety_checker = False)
    print("PEFT TYPE: " args.unet_pretraining_type)
    pipe, tunable_params = loadSDModel(args, args.output_dir)
    pipe = pipe.to("cuda")

    memorized_images, similar_images, stats = evaluate(args, pipe)

    if args.print_stats:
        print(f"Number of all images: {stats['total']}")
        print(f"Number of actual processed images: {stats['actual_total']}")
        print(f"Number of memorized images: {stats['memorized']}")
        print(f"Number of similar images: {stats['similar']}")
        print(f"Ratio of memorized images from all: {stats['memorized']}/{stats['total']}")
        print(f"Ratio of similar images from memorized: {stats['similar']}/{stats['memorized']}")

    if args.outputs_name:
        outputs = {
            "memorized_images": memorized_images,
            "similar_images": similar_images,
            "stats": stats
        }

        # make results directory if it does not exist
        results_dir = os.path.join(args.output_dir, "results_extraction_attack")
        os.makedirs(results_dir, exist_ok=True)

        results_filename = os.path.join(results_dir, args.outputs_name + ".json")

        with open(results_filename, "w") as f:
            json.dump(outputs, f)