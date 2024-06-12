import os
from natsort import natsorted
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from package1 import *

def init(save_path_prefix, log1_path, log2_path):
    # 创建npy保存路径
    os.makedirs(save_path_prefix, exist_ok=True)
    print("npy保存路径创建成功")
    
    # 创建日志csv文件
    if not os.path.isfile(log1_path):
        data = {
            'conversation': [],
            'sentence': [],
        }
        df = pd.DataFrame(data)
        df.to_csv(log1_path, index=False)
        print(f"{log1_path} 创建成功")
    else:
        df = pd.read_csv(log1_path)
        data = {
            'conversation': ['-------'],
            'sentence': ['-------'],
        }
        new_data = pd.DataFrame(data)
        print(new_data)
        df = pd.concat([df, new_data], ignore_index=True)
        print(df)
        df.to_csv(log1_path, index=False)
        
    if not os.path.isfile(log2_path):
        data = {
            'file': [],
            'frames': [],
        }
        df = pd.DataFrame(data)
        df.to_csv(log2_path, index=False)
        print(f"{log2_path} 创建成功")
    else:
        df = pd.read_csv(log2_path)
        data = {
            'file': ['-------'],
            'frames': ['-------'],
        }
        new_data = pd.DataFrame(data)
        print(new_data)
        df = pd.concat([df, new_data], ignore_index=True)
        print(df)
        df.to_csv(log2_path, index=False)

save_path_prefix = "/home/h666/Zq/MELD_dev_result"
log1_path = "/home/h666/Zq/MELD_dev_result/log1.csv"
log2_path = "/home/h666/Zq/MELD_dev_result/log2.csv"

init(save_path_prefix, log1_path, log2_path)





# 获取dev集的文件
dev_raw_data_prefix = "/home/h666/Zq/情绪识别数据集/MELD.Raw/dev_splits_complete"
dev_raw_data_files_name = os.listdir(dev_raw_data_prefix)
dev_raw_data_files_name = natsorted(dev_raw_data_files_name)

# 测试使用，只取前几个文件试试
# dev_raw_data_files_name = dev_raw_data_files_name[:200]

# 定义一个字典，用于存储训练集的分隔信息
dia = {}

# 分隔训练集文件
for file_name in dev_raw_data_files_name:
    # print(file_name)
    dia_begin = 3
    dia_end = file_name.index('_')
    dia_number = file_name[dia_begin:dia_end]
    
    utt_begin = dia_end + 4
    utt_end = file_name.index('.')
    utt_number = file_name[utt_begin:utt_end]
    
    # print(dia_number, " ", utt_number)
    
    if dia_number in dia:
        dia[dia_number].append(utt_number)
    else:
        dia[dia_number] = [utt_number]

# 手动删除不符合要求的视频tra
# dia.pop('134',"")
# dia['125'].remove('3')
dia['49'].remove('4')
dia['49'].remove('5')
dia['66'].remove('9')
dia['66'].remove('10')

# 测试使用，仅选择dia11
# dia = {key: value for key, value in dia.items() if key == '125'}

# 测试使用，检查字典
# dia = json.dumps(dia, indent=4)
# print(dia)


for dia_number in tqdm(dia, desc="Processing dia:"):
    features = []
    utt_number_sum = len(dia[dia_number])
    utt_number_valid_number = 0
    for utt_number in tqdm(dia[dia_number], desc=f"Processing dia{dia_number}:", leave=False):
        video_name = f"dia{dia_number}_utt{utt_number}.mp4"
        video_path = dev_raw_data_prefix + "/" + video_name
        
        result, feature = get_numpy_all(dev_raw_data_prefix, video_name, log2_path)

        if result is None:
            df = pd.read_csv(log1_path)
            new_data = {
                'conversation': ["dia" + dia_number],
                'sentence': ["utt" + utt_number],
            }
            new_data = pd.DataFrame(new_data)
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(log1_path, index=False)
            if feature is None:
                _, contrast_feature = get_numpy_all_whole(dev_raw_data_prefix, video_name, log2_path)
                features.append(contrast_feature)
        else:
            utt_number_valid_number += 1
            features.append(feature)
        
        del feature
    
    if utt_number_valid_number == 0:
        df = pd.read_csv(log1_path)
        new_data = {
            'conversation': ["dia" + dia_number],
            'sentence': ["all"],
        }
        new_data = pd.DataFrame(new_data)
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(log1_path, index=False)

        continue
    
    features = np.array(features)
    features = np.vstack(features)
    
    save_path = save_path_prefix + "/" + "dia" + dia_number + ".npy"
    np.save(save_path, features)
    
    del features

import gc
gc.collect()