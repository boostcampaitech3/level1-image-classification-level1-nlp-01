import torch
from loss import create_criterion
from model import CoralNet, CoralNet_EfficientNet, CoralNet_Pretrained
import json
import glob
import re
from pprint import pprint
import argparse
import numpy as np
import random
from pathlib import Path
from importlib import import_module
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import multiprocessing

NUM_CLASSES = 3
model_ = CoralNet_EfficientNet(NUM_CLASSES, False)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir
    )

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
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True
    )

    # -- model
    model = model_.to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    
    if args.lr_scheduler == 'StepLR':
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        
    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_loss = np.inf
    best_accuracy_model_path = None
    importance_weights = torch.ones(NUM_CLASSES-1, dtype=torch.float)
    importance_weights[1] = torch.tensor(3.0)
    
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, probas = model(inputs)
            logits = logits.to(device)
            probas = probas.to(device)
            importance_weights = importance_weights.to(device)

            loss = criterion(logits, labels, importance_weights)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)

                loss_value = 0

        scheduler.step()

        # val loop
        # phil
        
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits, probas = model(inputs)
                loss = criterion(logits, labels, importance_weights)
                val_loss_items.append(loss.cpu())
            
            val_loss = np.sum(val_loss_items) / len(val_loader)
            
            if val_loss < best_val_loss:
                if best_accuracy_model_path is not None:
                    os.remove(best_accuracy_model_path)
                best_val_loss = val_loss
                best_accuracy_model_path = f"{save_dir}/Epoch{epoch}_loss.pth"
                print(f"New best model for val loss : {val_loss:4.2}! saving the best model..")
                torch.save(model.module.state_dict(), best_accuracy_model_path)
                
            
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(f"[Val] loss: {val_loss:4.2} || best loss: {best_val_loss:4.2}")
            logger.add_scalar("Val/loss", val_loss, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='CoralMaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int,default=[224, 224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', help='learning scheduler (default: StepLR)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='coral', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='coral', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    pprint(vars(args))
    
    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)