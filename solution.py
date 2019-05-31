from argparse import ArgumentParser
import os
from PIL import Image
import numpy as np
from imagesignature import ImageSignature

np.warnings.filterwarnings('ignore')


def get_parser():
    parser = ArgumentParser(description="First test task on images similarity")
    parser.add_argument("--path", type=str, help="folder with images", default='.', required=True)

    return parser


def is_image_file(file):
    return file.is_file() and file.name.split('.')[-1].lower() in {'jpg', 'jpeg', 'png'}


def get_pairs(a):
    if len(a) <= 0:
        return []

    return [(a[0], el) for el in a[1:]] + get_pairs(a[1:])


if __name__ == "__main__":
    THRESHOLD = 0.61

    args = get_parser().parse_args()
    img_sig = ImageSignature()
    image_files = [entry for entry in os.scandir(args.path) if is_image_file(entry)]
    file_mapping = {i: file.name for i, file in enumerate(image_files)}
    signatures = dict()
    data = dict()

    assert (len(image_files) > 0), 'Oh! This directory is empty'

    # Compute signature for each image
    for index, file in enumerate(image_files):
        signatures[index] = img_sig.signature(Image.open(file.path))

    # Compare images
    pairs = get_pairs(list(file_mapping.keys()))

    for p in pairs:
        data[p] = img_sig.normalized_distance(signatures[p[0]], signatures[p[1]])

    # Print similar images
    for index, dist in sorted(data.items(), key=lambda k: k[1]):
        if dist <= THRESHOLD:
            print(file_mapping[index[0]], file_mapping[index[1]])
