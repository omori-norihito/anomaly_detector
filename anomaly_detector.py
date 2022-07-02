#!/usr/bin/env python3
import sys
import logging
import argparse
from pathlib import Path

sys.path.append('./ind_knn_ad/indad')

import torch
from torch import tensor
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)
method = "patchcore"
cls = "metal_nut"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])
size = 224
socore_threshold = 23

loader = transforms.Compose([
                             transforms.Resize(256,
                                               interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.CenterCrop(size),
                             transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                            ])


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def main(args):
    sample_img = image_loader(args.imagefile)
    model_path = Path(f'{method}_{cls}.pth')
    logger.debug(f"model_path: {model_path}")
    model = torch.load(model_path)
    model.eval()
    img_lvl_anom_score, pxl_lvl_anom_score = model.predict(sample_img)
    if img_lvl_anom_score.numpy() >= socore_threshold:
        print(f"{args.imagefile},{img_lvl_anom_score.numpy()},1")
    else:
        print(f"{args.imagefile},{img_lvl_anom_score.numpy()},0")


def parse_args():
    # オプションの解析
    parser = argparse.ArgumentParser(description='Anormaly detector of metal_nut')
    parser.add_argument(
                        'imagefile',
                        )
    parser.add_argument(
                        '-l', '--loglevel',
                        choices=('warning', 'debug', 'info'),
                        default='info'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.setLevel(args.loglevel.upper())
    logger.info('loglevel: %s', args.loglevel)
    lformat = '%(name)s <L%(lineno)s> [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=lformat,
    )
    logger.setLevel(args.loglevel.upper())

    main(args)
