import argparse
import os
from importlib import import_module
import logging
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import utils
from pprint import pprint

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model.load_state_dict(torch.load(saved_model, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_num_class = getattr(import_module("dataset"), args.num_dataset)
    
    num_classes = dataset_num_class.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(img_paths, args.resize)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
#         num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    log_logger.info("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    exp_name = output_dir.split('/')[-1]
    info.to_csv(os.path.join(output_dir, f'{exp_name}_output.csv'), index=False)
    log_logger.info(f'== Inference Done! ==')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--dataset', type=str, default='TestDataset', help='dataset augmentation type (default: TestDataset)')
    parser.add_argument('--num_dataset', type=str, default='OnlyAgeMaskSplitByProfileDataset', help='number of classes used for dataset')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--config', default='./configs/model_config.json', help='config.json file')
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_path', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_path', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()    
    config = utils.read_json(args.config)
    parser.set_defaults(**config['inference'])
    args = parser.parse_args()
    
    data_dir = args.data_dir
    model_dir = args.model_path
    output_dir = args.output_path
    
    # Setting up Logger
    log_logger = logging.getLogger(__name__)
    utils.setup_logging(output_dir, __file__)
    log_logger.info('Inference Parameters') 
    log_logger.info(json.dumps(vars(args), indent=4))
    log_logger.info('== Start Inference ==')

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
