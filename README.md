# HSE提取体特征

## 原理

- 对于一张图片，首先使用Github源代码中给的preprocess函数，对图片进行缩放，然后使用MTCNN的模型(具体实施方式为使用Github源代码中的类`FacialImageProcessing`中的`detect_faces`函数），检测出人脸的位置，用两个坐标限定，如果有多个人脸，则返回多个人脸的位置信息，最后对人脸区域进行对齐（将检测出的人脸区域缩放为正方形）
- 然后使用类`HSEmotionRecognizer`（在hsemotion中的facial_emotions，具体看代码）中的`extract_features`函数，直接抽取特征

## src解释

- `facial_analysis.py`为Github源代码中检测分析人脸的代码，`package1.py`是为了处理数据集特征而写的代码，里面写了提取单个视频特征的函数
- `test_extract_feature_from_picture.py`为测试一下当前的环境是否可以检测图片中的人脸并提取特征

## 使用步骤

此处以MELD_train_process.py为例

### 环境

- 推荐使用conda环境，然后使用官方给的requirements.txt进行安装

    ```shell
    conda create -n HSE
    conda activate HSE
    
    # 进入项目根目录
    pip install -r requirements.txt
    
    # 如果遇到dlib报错，可以先跳过，因为这里用不到，或者使用conda下载dlib
    ```

- 安装人脸分析的包

    ```shell
    pip install hsemotion
    ```

### 确定路径

1. 数据集的路径
2. 处理结果保存的路径

### 书写处理代码

1. 复制`MELD_train_process.py`的内容

2. 修改`save_path_prefix`为处理结果保存的路径

3. 修改`log1_path`为处理结果的日志文件1的保存路径

4. 修改`log2_path`为处理结果的日志文件2的保存路径

5. 修改`train_raw_data_prefix`为数据集的视频文件的父目录

6. 所有要处理的文件的信息都保存在`dia`中，根据需求，手动删除不符合规矩的视频

    ```python
    # 删除dia134的所有视频
    dia.pop('134',"")
    
    # 仅删除dia125中的视频utt3
    dia['125'].remove('3')
    ```

### 注意事项

- `package1.py`中指定了使用第几张GPU

    ```python
    # 指定使用第二个GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    ```

- `MELD`数据集的视频是放在同一个目录下面的，如果某个数据集的视频是放在许多不同的目录下面，那么需要修改一下代码，让字典`dia`存储所有数据集的信息，其中键为对话的序号，值为语句的序号
- `log1.csv`中存储不符合要求的视频文件，主要是两种
    1. 视频文件为空
    2. 视频文件中不存在一张能检测到人脸的帧

- `log2.csv`中存储提取视频特征的记录，包括文件名和提取的帧数