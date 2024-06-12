import os
from natsort import natsorted
import pandas as pd
import numpy as np
from package1 import get_numpy_all
from tqdm import tqdm

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

directories_prefix = "/home/h666/Zq/IEMOCAPvideo"
save_path_prefix = "/home/h666/Zq/IEMOCAPvideo_result"
log1_path = "/home/h666/Zq/IEMOCAPvideo_result/log1.csv"
log2_path = "/home/h666/Zq/IEMOCAPvideo_result/log2.csv"
directories = os.listdir(directories_prefix)
directories = natsorted(directories)

init(save_path_prefix, log1_path, log2_path)

for dir in tqdm(directories, desc="Processing directories"):
    save_path = os.path.join(save_path_prefix, dir + ".npy")
    features = []
    
    files = natsorted(os.listdir(os.path.join(directories_prefix, dir)))
    for file in tqdm(files, desc=f"Processing files in {dir}", leave=False):
        video_path = os.path.join(directories_prefix, dir, file)
        result, feature = get_numpy_all(directories_prefix + "/" + dir, file, log2_path)

        if result is None:
            df = pd.read_csv(log1_path)
            new_data = {
                'conversation': [dir],
                'sentence': [file],
            }
            new_data = pd.DataFrame(new_data)
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(log1_path, index=False)
        else:
            features.append(feature)
    
    if not features:
        print(dir, " : conversation is empty")
        df = pd.read_csv(log1_path)
        new_data = {
            'conversation': [dir],
            'sentence': ["all"],
        }
        new_data = pd.DataFrame(new_data)
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(log1_path, index=False)
        continue
    
    features = np.array(features)
    features = np.vstack(features)
    np.save(save_path, features)

import gc
gc.collect()
