from typing import Iterator
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from PIL import Image
import PIL
from glob import glob
from os import path
from tqdm import tqdm

class RandomBatchwiseSampler(Sampler):
    """
    Samples elements randomly, maintaining that elements are kept in the same
    relative batches. In other words, shuffles between batches.

    use like: DataLoader(ragged_image_dataset, batch_sampler=RandomBatchwiseSampler(len(ragged_image_dataset), batch_size))
    """
    def __init__(self, _len:int, batch_size: int):
        self.n_batches = _len // batch_size + (0 if _len % batch_size == 0 else 1)
        self._len = _len
        self.batch_size = batch_size

    def __iter__(self):
        for batch_i in torch.randperm(self.n_batches):
            start_i = batch_i.item()*self.batch_size
            idxs = torch.arange(start_i, min(self._len, start_i + self.batch_size))
            yield idxs

    def __len__(self) -> int:
        return self.n_batches

class RaggedImageDataset(Dataset):
    def __init__(
        self,
        image_path,
        batch_size=16,
        largest_side_res=512,
        smallest_side_res=64,
        max_aspect_ratio=10.0,
        min_aspect_ratio=0.2,
        ext=".jpg",
        bucketing_metric=torch.median,
        post_resize_transforms=[transforms.ToTensor(), transforms.Normalize(0,1)],
        scale_factor=1.0,
    ):
        """
        Returns tensor batches of equal shape, from underlying images with different aspect ratios.

        Does a first pass over all images in path/*ext, getting their dimensions and checking that they are not corrupt.

        Then sorts the images by aspect ratio, and then groups together images into batches, assigning to them
            widths and heights based on bucketing_metric
        """

        image_files = glob(path.join(image_path, f"**/*{ext}"), recursive=True)
        widths, heights = get_images_dimensions(image_files)

        error_indeces = torch.where(widths == 0)[0]
        if len(error_indeces) > 0:
            print(
                "Warning!, the aspect ratios of the following images were unable to be read:\n"
            )
            print("".join([f"\t'{image_files[i]}'\n" for i in error_indeces]))

        aspect_ratios = widths / heights
        aspect_ratios, aspect_ratios_indeces = aspect_ratios.sort()
        _mask = torch.logical_and(
            torch.logical_and(aspect_ratios != 0, min_aspect_ratio < aspect_ratios),
            aspect_ratios < max_aspect_ratio,
        )
        aspect_ratios_indeces = aspect_ratios_indeces[_mask]
        widths = widths[aspect_ratios_indeces]
        heights = heights[aspect_ratios_indeces]
        widths, heights = clamp_by_min_res(widths, heights, smallest_side_res)
        widths, heights = clamp_by_max_res(widths, heights, largest_side_res)
        self.aspect_ratios = widths / heights
        self.widths = widths
        self.heights = heights

        self.image_files = [image_files[i] for i in aspect_ratios_indeces]

        indeces = torch.arange(self.widths.shape[0])
        split_indeces = torch.split(indeces, batch_size)

        width_buckets = torch.IntTensor(
            [bucketing_metric(self.widths[si]) for si in split_indeces]
        )
        height_buckets = torch.IntTensor(
            [bucketing_metric(self.heights[si]) for si in split_indeces]
        )
        bucketed_indeces = indeces // batch_size

        self.bucketed_widths = width_buckets[bucketed_indeces]
        self.bucketed_heights = height_buckets[bucketed_indeces]
        self.bucketed_aspect_ratios = self.bucketed_widths / self.bucketed_heights

        self.post_resize_transforms = post_resize_transforms
        self.scale_factor = scale_factor

    def __getitem__(self, idx):
        """
        returns image, filename
        """
        width, height = self.bucketed_widths[idx], self.bucketed_heights[idx]
        # by default, scale_factor is 1.0
        width = int(self.scale_factor * width)
        height = int(self.scale_factor * height)
        image_transforms = transforms.Compose([transforms.Resize((height, width,)), *self.post_resize_transforms])
        try:
            with Image.open(self.image_files[idx]) as im:
                im = image_transforms(im)
        except Exception as e:
            print(self.image_files[idx])
            print(e)
            return None
        return im, self.image_files[idx]


    def __len__(self):
        return len(self.image_files)

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)


def clamp_by_max_res(widths, heights, largest_side_res):
    widths = widths.clone()
    heights = heights.clone()
    widths, heights = _clamp_by_max(widths, heights, largest_side_res)
    heights, widths = _clamp_by_max(heights, widths, largest_side_res)
    return widths, heights


def clamp_by_min_res(widths, heights, smallest_side_res):
    widths = widths.clone()
    heights = heights.clone()
    widths, heights = _clamp_by_min(widths, heights, smallest_side_res)
    heights, widths = _clamp_by_min(heights, widths, smallest_side_res)
    return widths, heights


def _clamp_by_min(a, b, _min):
    smaller_a = a <= b
    adjusted_widths = torch.clamp_min(a[smaller_a], _min)
    grow_ratio = adjusted_widths / a[smaller_a]
    a[smaller_a] = adjusted_widths
    b_mod = b[smaller_a] * grow_ratio
    b[smaller_a] = torch.round(b_mod).to(b.dtype)
    return a, b


def _clamp_by_max(a, b, _max):
    bigger_a = a >= b
    adjusted_widths = torch.clamp_max(a[bigger_a], _max)
    shrink_ratio = adjusted_widths / a[bigger_a]
    a[bigger_a] = adjusted_widths
    b_mod = b[bigger_a] * shrink_ratio
    b[bigger_a] = torch.round(b_mod).to(b.dtype)
    return a, b


def get_images_dimensions(image_files):
    """
    If the image fails to load, sets the width to 0 and height to -1
    """
    widths = torch.zeros(len(image_files), dtype=torch.int)
    heights = torch.zeros(len(image_files), dtype=torch.int)
    for i, image_file in tqdm(enumerate(image_files), desc="loading image dimensions", total=len(image_files)):
        try:
            with Image.open(image_file) as im:
                width, height = im.size
                mode = im.mode
        except PIL.UnidentifiedImageError:
            widths[i] = 0
            heights[i] = -1
            continue
        if mode!="RGB":
            widths[i] = 0
            heights[i] = -1
            continue
        widths[i] = width
        heights[i] = height
    return widths, heights
