import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import utils
from pprint import pprint
from dataset import MaskBaseDataset

def load_model(saved_model, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls()

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # model_path = os.path.join(saved_model, 'Epoch40_accuracy.pth')
    model.load_state_dict(torch.load(saved_model, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    
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

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            out_age, out_mask, out_sex = model(images)

            preds_age = torch.argmax(out_age, dim=-1)
            preds_mask = torch.argmax(out_mask, dim=-1)
            preds_sex = torch.argmax(out_sex, dim=-1)

            preds += [MaskBaseDataset.encode_multi_class(preds_mask[i], preds_sex[i], preds_age[i]) for i in range(len(preds_age))]
            
    preds = torch.stack(preds)
    preds = preds.cpu().numpy()    
    info['ans'] = preds
    exp_name = output_dir.split('/')[-1]
    info.to_csv(os.path.join(output_dir, f'{exp_name}_output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--dataset', type=str, default='TestDataset', help='dataset augmentation type (default: TestDataset)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--config', default='./configs/best_model_config.json', help='config.json file')
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_path', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_path', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()
    
    config = utils.read_json(args.config)
    args = utils.update_argument(args, config['inference'])
    pprint(vars(args))
    
    data_dir = args.data_dir
    model_dir = args.model_path
    output_dir = args.output_path

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
