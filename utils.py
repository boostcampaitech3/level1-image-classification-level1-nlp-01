import json
from importlib import import_module
# import torch

def update_argument(args, configs):
    for arg in configs:
        if arg in args:
            setattr(args, arg, configs[arg])
        else:
            raise ValueError(f"no argument {arg}")
    return args

def read_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data

def write_json(data, file, indent=4):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

