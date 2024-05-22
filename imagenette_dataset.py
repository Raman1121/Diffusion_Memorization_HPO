import torch
from PIL import Image
import random
import os
import pandas as pd
from pathlib import Path
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


def insert_rand_word(sentence, word):
    sent_list = sentence.split(" ")
    sent_list.insert(random.randint(0, len(sent_list)), word)
    new_sent = " ".join(sent_list)
    return new_sent


def prompt_augmentation(
    prompt, aug_style="rand_word_add", tokenizer=None, repeat_num=4
):
    if aug_style == "rand_numb_add":
        for i in range(repeat_num):
            randnum = np.random.choice(100000)
            prompt = insert_rand_word(prompt, str(randnum))
    elif aug_style == "rand_word_add":
        for i in range(repeat_num):
            rand_int = list(np.random.randint(49400, size=1))
            randword = tokenizer.decode(rand_int)
            prompt = insert_rand_word(prompt, randword)
    elif aug_style == "rand_word_repeat":
        wordlist = prompt.split(" ")
        for i in range(repeat_num):
            randword = np.random.choice(wordlist)
            prompt = insert_rand_word(prompt, randword)
    else:
        raise Exception("This style of prompt augmnentation is not written")
    return prompt


class ImagenetteDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        csv_file,
        root_path,
        tokenizer=None,
        transform=None,
        classifier_guidance_dropout=0.1,
        use_random_word_addition=False,
    ):
        self.csv_file = csv_file
        self.root_path = root_path
        self.tokenizer = tokenizer
        self.transform = transform
        self.classifier_guidance_dropout = classifier_guidance_dropout
        self.use_random_word_addition = use_random_word_addition

        self.df = pd.read_csv(csv_file)

        if self.tokenizer is not None:
            # RWA
            if self.use_random_word_addition:
                self.df["text"] = self.df["text"].apply(
                    lambda x: prompt_augmentation(x, tokenizer=self.tokenizer)
                )

            self.tokens = self.tokenizer(
                self.df["text"].to_list(),
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
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.root_path / self.df["path"].iloc[idx]

        try:
            im = Image.open(img_path).convert("RGB")
        except:
            print("ERROR IN LOADING THE IMAGE {}".format(img_path))

        if self.transform:
            im = self.transform(im)

        text = self.df["text"].iloc[idx]
        if self.use_random_word_addition:
            text = prompt_augmentation(text, tokenizer=self.tokenizer)

        sample = {
            "image": im,
            "text": text,
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
