import torch
from PIL import Image
import random
import os
import pandas as pd
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MimicCXRDataset(torch.utils.data.Dataset):
    """Mimic CXR dataset."""

    def __init__(
        self,
        images_dir,
        tokenizer=None,
        csv_file: Path = None,
        transform=None,
        seed=42,
        classifier_guidance_dropout=0.1,
        dataset_size_ratio=None,
        use_real_images: bool = True,
        use_findings: bool = False
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            images_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.classifier_guidance_dropout = classifier_guidance_dropout
        self.use_findings = use_findings

        random.seed(seed)

        if(isinstance(csv_file, pd.DataFrame)):
            # We can either pass the dataframe directly
            self.annotations_text_image_path = csv_file
        else:
            # Or pass the path to the dataframe
            try:
                self.annotations_text_image_path = pd.read_excel(csv_file)
            except:
                self.annotations_text_image_path = pd.read_excel(csv_file)

        if not use_real_images:
            self.img_path_key = "synth_img_path"
            self.annotations_text_image_path = get_synthetic_df(
                self.annotations_text_image_path, images_dir
            )
        else:
            self.img_path_key = "path"

        if dataset_size_ratio is not None:
            original_dataset_size = len(self.annotations_text_image_path)
            dataset_size = int(
                len(self.annotations_text_image_path) * dataset_size_ratio
            )
            subset_rows = random.sample(range(original_dataset_size), k=dataset_size)
            # subset_rows = random.sample(range(dataset_size), k=dataset_size)
            # self.annotations_text_image_path = self.annotations_text_image_path.iloc[:dataset_size]
            self.annotations_text_image_path = self.annotations_text_image_path.iloc[
                subset_rows
            ]

        if(self.use_findings):
            assert all(
                [
                    isinstance(text, str)
                    for text in self.annotations_text_image_path["findings"].to_list()
                ]
            ), "All text must be strings"
        else:
            assert all(
                [
                    isinstance(text, str)
                    for text in self.annotations_text_image_path["text"].to_list()
                ]
            ), "All text must be strings"

        if self.tokenizer is not None:
            if(self.use_findings):
                self.tokens = self.tokenizer(
                    self.annotations_text_image_path["findings"].to_list(),
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                )
            else:
                self.tokens = self.tokenizer(
                    self.annotations_text_image_path["text"].to_list(),
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                )
            self.uncond_tokens = self.tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

    def __len__(self):
        return len(self.annotations_text_image_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = (
            self.images_dir
            / self.annotations_text_image_path[self.img_path_key].iloc[idx]
        )

        try:
            im = Image.open(img_path).convert("RGB")
        except:
            print("ERROR IN LOADING THE IMAGE {}".format(img_path))
        if self.transform:
            im = self.transform(im)

        if(self.use_findings):
            sample = {
                "image": im,
                "text": self.annotations_text_image_path["findings"].iloc[idx],
            }
        else:
            sample = {
                "image": im,
                "text": self.annotations_text_image_path["text"].iloc[idx],
            }

        if self.tokenizer is not None:
            if random.randint(0, 100) / 100 < self.classifier_guidance_dropout:
                input_ids, attention_mask = torch.LongTensor(
                    self.uncond_tokens.input_ids
                ), torch.LongTensor(self.uncond_tokens.attention_mask)
            else:
                input_ids, attention_mask = torch.LongTensor(
                    self.tokens.input_ids[idx]
                ), torch.LongTensor(self.tokens.attention_mask[idx])
            sample["input_ids"] = input_ids
            sample["attention_mask"] = attention_mask

        return sample


def get_synthetic_df(
    df: pd.DataFrame, synthetic_images_path: Path, chexpert_labels_path: Path = None
):
    if "img_name" not in df.columns:
        df["img_name"] = df["path"].map(lambda x: x[x.rfind("/") + 1 : x.rfind(".")])
    if "synth_img_path" not in df.columns:
        imgs_path_list = [str(i.name) for i in synthetic_images_path.glob("*")]
        df_synth = pd.DataFrame(columns=["synth_img_path"], data=imgs_path_list)
        df_synth["img_name"] = df_synth["synth_img_path"].map(
            lambda x: x[: x.find("_")]
        )
        df = pd.merge(df_synth, df, how="left", on="img_name")

    if chexpert_labels_path is not None:
        df_chexpert = pd.read_csv(chexpert_labels_path)
        df = pd.merge(
            df,
            df_chexpert.rename(columns={"study_id": "study"}),
            how="left",
            on=["subject_id", "study"],
        )

    return df
