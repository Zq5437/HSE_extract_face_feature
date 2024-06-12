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

save_path_prefix = "/home/h666/Zq/MELD_test_result"
log1_path = "/home/h666/Zq/MELD_test_result/log1.csv"
log2_path = "/home/h666/Zq/MELD_test_result/log2.csv"

init(save_path_prefix, log1_path, log2_path)





# 获取测试集的文件
test_raw_data_prefix = "/home/h666/Zq/情绪识别数据集/MELD.Raw/output_repeated_splits_test"
test_raw_data_files_name = os.listdir(test_raw_data_prefix)

test_raw_data_files_name = [file for file in test_raw_data_files_name if file.endswith(".mp4")]
test_raw_data_files_name = [file for file in test_raw_data_files_name if file.startswith("dia")]

test_raw_data_files_name = natsorted(test_raw_data_files_name)

# 测试使用，只取前几个文件试试
# test_raw_data_files_name = test_raw_data_files_name[:200]

# 定义一个字典，用于存储训练集的分隔信息
dia = {}

# 分隔训练集文件
for file_name in test_raw_data_files_name:
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

# 手动删除不符合要求的视频
# dia.pop('134',"")
# dia['125'].remove('3')
dia['220'].remove('0')
dia['220'].remove('1')
dia['93'].remove('5')
dia['93'].remove('6')
dia['93'].remove('7')
dia['108'].remove('1')
dia['108'].remove('2')


# 测试使用，仅选择dia220
# dia = {key: value for key, value in dia.items() if key == '220'}


# 程序中断后，从断点处继续
# break_point = '220'
# dia = {key: value for key, value in dia.items() if int(key) >= int(break_point)}


# 测试使用，检查字典
# dia = json.dumps(dia, indent=4)
# print(dia)

for dia_number in tqdm(dia, desc="Processing dia:"):
    features = []
    utt_number_sum = len(dia[dia_number])
    utt_number_valid_number = 0
    for utt_number in tqdm(dia[dia_number], desc=f"Processing dia{dia_number}:", leave=False):
        video_name = f"dia{dia_number}_utt{utt_number}.mp4"
        video_path = test_raw_data_prefix + "/" + video_name
        
        result, feature = get_numpy_all(test_raw_data_prefix, video_name, log2_path)

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
                _, contrast_feature = get_numpy_all_whole(test_raw_data_prefix, video_name, log2_path)
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