"""
usage:
    python filter.py -f <name of image folder> --kwlist <keyword 1>, <key word 2>, .... --thresh 0.9

Will copy matched files to the keywords to a subfolder called filtered
"""

import argparse
from shutil import copyfile
from typing import List
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm

from dataset import RaggedImageDataset, RandomBatchwiseSampler


def main(folder: str, kw_list: List[str], thresh:float, clipmodel:str, batch_size=32):
    _outfolder = "filtered/"
    os.makedirs(os.path.join(folder, _outfolder), exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CLIPModel.from_pretrained(clipmodel).to(device)
    model = model.eval()
    processor = CLIPProcessor.from_pretrained(clipmodel)

    dataset = RaggedImageDataset(
        folder,
        batch_size,
        largest_side_res=512,
        smallest_side_res=256,
        ext=".jpg",
        post_resize_transforms=[
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ],
    )
    dataloader = DataLoader(
        dataset, batch_sampler=RandomBatchwiseSampler(len(dataset), batch_size)
    )
    input_ids = torch.Tensor(processor(text=kw_list, padding=True).input_ids).to(torch.int).to(device)
    text_embeds = model.get_text_features(input_ids)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    for image_b, filename_b in tqdm(dataloader):
        image_b = image_b.to(device)
        image_embeds = model.get_image_features(pixel_values=image_b)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        probs_per_image = logits_per_image.softmax(dim=1)
        matched_per_image = probs_per_image > thresh

        for filename, matched_list in zip(filename_b, matched_per_image):
            for kw, was_matched in zip(kw_list, matched_list):
                if not was_matched:
                    continue
                bn = os.path.basename(filename)
                dst = os.path.join(folder, _outfolder) + kw + "-" + bn
                copyfile(filename, dst)


if __name__ in {"__console__", "__main__"}:
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", dest="folder", help="folder")
    ap.add_argument("--kwlist", nargs="+", dest="kwlist", help="key word list for clip")
    ap.add_argument("--thresh", type=float, dest="thresh", help="threshold for matching key word", default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--clipmodel", type=str, default="openai/clip-vit-base-patch32")
    args = ap.parse_args()
    with torch.no_grad():
        main(args.folder, args.kwlist, args.thresh, args.clipmodel, batch_size=args.batch_size)
