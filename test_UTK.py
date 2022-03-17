import argparse
import os
from importlib import import_module

import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset2, MaskBaseDataset
import utils
from pprint import pprint

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes = num_classes
    )

    model.load_state_dict(torch.load(saved_model, map_location=device))

    return model

def UTK_label(gender, age): #age는 label값이 아닌 실제 age값이 들어오게 됨.
    gender = int(gender)
    age = int(age)
    if age < 30:
        age = 0
    elif 30 <= age and age < 60:
        age = 1
    else:
        age = 2

    return gender * 3 + age

def get_answers(data_dir):
    img_paths = sorted(os.listdir(data_dir))
    answers = []
    for img_path in img_paths:
        age, gender, race, id = img_path.split("_")
        if race == '2':
            answers.append(UTK_label(gender, age))
    
    return answers

# def f1_loss(preds, trues):


@torch.no_grad()
def test(data_dir, model_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = "./UTKFace"
    img_paths = [os.path.join(img_root, img_path) for img_path in sorted(os.listdir(data_dir))]
    dataset = TestDataset2(img_paths, args.resize, mean=args.mean, std=args.std)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False
    )

    print("Calculation test results(with UTKFace images)...")
    preds = []
    answers = get_answers(data_dir) 
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred[1].argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    raw_data = {
        'img_paths' : img_paths,
        'answers' : answers,
        'preds' : preds
    }
    df = DataFrame(raw_data)
    df.to_csv("./inference_UTK/info3.csv", header=True, index=False)

    cnt = 0
    for i in range(len(answers)):
        if preds[i] != answers[i]:
            cnt += 1

    print(f"test acc : {((1-cnt/len(answers)) * 100):.2f}%")
    # print(f"test f1 score: {f1_loss}")
    print("Test Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--config', default='./configs/model_config.json', help='config.json file')
    parser.add_argument('--mean', default=[0.548, 0.504, 0.479], help= "mean")
    parser.add_argument('--std', default=[0.237, 0.247, 0.246], help='std')
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_path', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_path', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './inference'))

    args = parser.parse_args()
    
    config = utils.read_json(args.config)
    args = utils.update_argument(args, config['test'])
    pprint(vars(args))
    
    data_dir = args.data_dir
    model_dir = args.model_path

    test(data_dir, model_dir, args)


    