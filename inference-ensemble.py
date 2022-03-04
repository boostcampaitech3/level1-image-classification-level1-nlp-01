import argparse
import os
import pandas as pd
from dataset import TestDataset, MaskBaseDataset


def inference(output_dir, args):

    df_age = pd.read_csv(args.age_inference)
    df_mask = pd.read_csv(args.mask_inference)
    df_gender = pd.read_csv(args.gender_inference)
    
    age_pred = df_age['ans'].tolist()
    mask_pred = df_mask['ans'].tolist()
    gender_pred = df_gender['ans'].tolist()

    pred = []
    dataset = MaskBaseDataset
    
    for age, mask, gender in zip(age_pred, mask_pred, gender_pred):
        label = dataset.encode_multi_class(mask, gender, age)
        pred.append(label)

    df_age['ans'] = pred
    df_age.to_csv(os.path.join(output_dir, f'41_43_45_output.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--age_inference', type=str, default='./model/exp41/exp41_output.csv')
    parser.add_argument('--mask_inference', type=str, default='./model/exp43/exp43_output.csv')
    parser.add_argument('--gender_inference', type=str, default='./model/exp45/exp45_output.csv')
    
    parser.add_argument("--output_path", default='./inference_ensemble')
    

    args = parser.parse_args()
    output_dir = args.output_path
  
    os.makedirs(output_dir, exist_ok=True)
    inference(output_dir, args)
