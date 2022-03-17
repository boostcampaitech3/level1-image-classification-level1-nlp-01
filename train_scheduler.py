import os, sys
import utils
import json

def set_config(configs, arg_name, arg_value):
    configs[arg_name] = arg_value

train_config_base = utils.read_json('./configs/model_config_base.json')
train_config = utils.read_json('./configs/model_config_base.json')
schedule = utils.read_json("./train_schedule.json")

# 'train_schedule.json'에 적힌대로 매 실험마다 model_config.json 변경하여 train.py 실행.
for num in schedule:
    print(f"training ({num}) ...")
    for arg in schedule[num]:
        set_config(train_config['train'], arg, schedule[num][arg])
    
    utils.write_json(train_config, './configs/model_config.json')
    os.system(f"python train.py")
    print(f"training ({num}) is done", end='\n\n')
    utils.write_json(train_config_base, './configs/model_config.json')
