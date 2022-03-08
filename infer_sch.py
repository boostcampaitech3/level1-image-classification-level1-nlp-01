from pprint import pprint
import os

def run():
    file_lst = ['/opt/ml/baseline_code_1_phil/configs/model_res_gender_config.json',
               '/opt/ml/baseline_code_1_phil/configs/model_res_age_config.json',
               '/opt/ml/baseline_code_1_phil/configs/model_res_mask_config.json']
    
    for file in file_lst:
        cmd = f"python inference.py --config {file}"
        os.system(cmd)
    
if __name__ == "__main__":
    run()