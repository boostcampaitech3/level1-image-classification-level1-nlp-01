import os
import logging.config 
import json
from importlib import import_module
import torch


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

def setup_logging(output_path, script_name,
                  default_path='./logs/logging.json', default_level=logging.INFO):
    """
        Setup logging configuration
    """
        
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        log_file_path = os.path.join(output_path, f'{script_name}' + '.logs')
        config["handlers"]["info_file_handler"]["filename"] = log_file_path
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
