# HSE Extracting Body Features

## Principle

- For an image, first use the preprocess function provided in Github source code to scale the image, then use the MTCNN model (specific implementation method is to use the `detect_faces` function in the class `FacialImageProcessing` in Github source code) to detect the position of the face, define it with two coordinates, and if there are multiple faces, return the position information of multiple faces. Finally, align the face regions (scale the detected face regions to squares)
- Then use the `extractfeatures` function in the class `HSEmotionRecognizer` (facial_emotion in hsemotion, depending on the code) to directly extract features

## Src explanation

- `facial_analysis.py` is the code for detecting and analyzing faces in Github source code, while `package1.py` is the code written to process dataset features, which includes functions for extracting individual video features
- To test whether the current environment can detect faces in images and extract features,  `test_extract_feature_from_picture.py`

## Usage steps

Taking MELD_train_process.py as an example here

### environment

- It is recommended to use the Conda environment and then use the official requirements. txt for installation

```Shell
conda create -n HSE
conda activate HSE
    
# Enter the project root directory
pip install -r requirements.txt
    
# If encountering dlib error, you can skip it first because it is not needed here, or use Conda to download dlib
```

- Install a facial analysis package

```Shell
pip install hsemotion
```

### Determine the path

1. Path to the dataset
2. Path for saving processing results

### Writing processing code

1. Copy the content of  `MELD_train_process.py`

2. Modify `save_path_prefix` to the path where the processing results are saved

3. Modify `log1 path` to the save path of log file 1 for processing results

4. Modify `log2_path` to the save path of log file 2 for processing results

5. Modify `train_raw_data_prefix` to the parent directory of the video file in the dataset

6. All information about the files to be processed is saved in `dia`, and videos that do not comply with the rules are manually deleted according to needs

```Python
# Delete all videos of dia134
dia.pop ('134', "")
    
# Only delete video utt3 in dia125
dia ['125'].remove ('3')
```

### Precautions

- The `package1.py` specifies which GPU to use

```Python
# Specify the use of the second GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
```

- The videos of the MELD dataset are placed in the same directory. If the videos of a dataset are placed in many different directories, the code needs to be modified so that the dictionary `dia` stores information for all datasets, where the key is the dialogue number and the value is the statement number
- There are two main types of video files stored in `log1. csv` that do not meet the requirements
    1. The video file is empty
    1. There is no frame in the video file that can detect faces


- Store records of extracting video features in `log2. csv`, including file names and extracted frame rates