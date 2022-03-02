import argparse
import os
import random
from importlib import import_module
import multiprocessing

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import utils
import dataset

from pprint import pprint
from sklearn.metrics import f1_score
import numpy as np


def load_model(saved_model, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls()

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # model_path = os.path.join(saved_model, 'Epoch40_accuracy.pth')
    model.load_state_dict(torch.load(saved_model, map_location=device))

    return model


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

def collate_fn(batch):
    
    data_list, label_list = [], []
    
    for _data, _label in batch:
        data_list.append(_data)
        label_list.append(torch.LongTensor(_label))
    data_list = torch.stack(data_list, dim = 0)
    label_list = torch.stack(label_list, dim = 1)
    
    return torch.Tensor(data_list), label_list


def validation(data_dir, model_dir, output_dir, args):
    """
        Calculate Accuracy and F1 Score on Validation Dataset
    """
    seed_everything(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        data_dir=args.data_dir,
    )
    num_classes = dataset.num_classes #18
    
    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()
    
    train_loader = DataLoader(
        train_set,
        collate_fn = collate_fn,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_set,
        collate_fn = collate_fn,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
        
    model = load_model(model_dir, device).to(device)
    model.eval()

    print("Calculating Validation results")
    f1_pred = []
    f1_labels = []
    train_acc_items = []
    val_acc_items = []
    
    with torch.no_grad():
        for train_batch in train_loader:
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            out_age, out_mask, out_sex = model(inputs)

            preds_age = torch.argmax(out_age, dim=-1)
            preds_mask = torch.argmax(out_mask, dim=-1)
            preds_sex = torch.argmax(out_sex, dim=-1)

            age_labels = labels[0,:]
            mask_labels = labels[1,:]
            sex_labels = labels[2,:]

            f1_pred += [dataset.encode_multi_class(preds_mask[i], preds_sex[i], preds_age[i]) for i in range(len(preds_age))]
            f1_labels += [dataset.encode_multi_class(mask_labels[i], sex_labels[i], age_labels[i]) for i in range(len(preds_age))]

            acc_item = ((preds_age == age_labels) & (preds_mask == mask_labels) & (preds_sex == sex_labels)).sum().item()
            train_acc_items.append(acc_item)

        f1_labels = torch.stack(f1_labels)
        f1_labels = f1_labels.cpu().numpy()
        f1_pred = torch.stack(f1_pred)
        f1_pred = f1_pred.cpu().numpy()
        f1_macro = f1_score(f1_labels, f1_pred, average='macro')

        train_acc = np.sum(train_acc_items) / len(train_set)
        print(
            f"[Train] acc : {train_acc:6.6%} ||"
            f"[Train] f1_macro : {f1_macro:6.6%}"
        )
        
        f1_pred = []
        f1_labels = []
        for val_batch in val_loader:
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            out_age, out_mask, out_sex = model(inputs)

            preds_age = torch.argmax(out_age, dim=-1)
            preds_mask = torch.argmax(out_mask, dim=-1)
            preds_sex = torch.argmax(out_sex, dim=-1)                                   

            age_labels = labels[0,:]
            mask_labels = labels[1,:]
            sex_labels = labels[2,:]

            f1_pred += [dataset.encode_multi_class(preds_mask[i], preds_sex[i], preds_age[i]) for i in range(len(preds_age))]
            f1_labels += [dataset.encode_multi_class(mask_labels[i], sex_labels[i], age_labels[i]) for i in range(len(preds_age))]

            acc_item = ((preds_age == age_labels) & (preds_mask == mask_labels) & (preds_sex == sex_labels)).sum().item()
            val_acc_items.append(acc_item)

        f1_labels = torch.stack(f1_labels)
        f1_labels = f1_labels.cpu().numpy()
        f1_pred = torch.stack(f1_pred)
        f1_pred = f1_pred.cpu().numpy()
        f1_macro = f1_score(f1_labels, f1_pred, average='macro')

        val_acc = np.sum(val_acc_items) / len(val_set)
        print(
            f"[Val] acc : {val_acc:6.6%} ||"
            f"[Val] f1_macro : {f1_macro:6.6%}"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='model type (default: BaseModel)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--config', default='./configs/best_model_config.json', help='config.json file')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_path', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_path', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()
    
    config = utils.read_json(args.config)
    args = utils.update_argument(args, config["valid"])
    pprint(vars(args))
    
    data_dir = args.data_dir # /opt/ml/input/data/train/images
    model_path = args.model_path # ./model/exp/Epoch40_accuracy.pth
    output_path = args.output_path # ./model/exp
    
    os.makedirs(output_path, exist_ok=True)

    validation(data_dir, model_path, output_path, args)