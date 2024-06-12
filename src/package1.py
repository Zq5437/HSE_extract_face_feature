from hsemotion import facial_emotions
from facial_analysis import FacialImageProcessing
from matplotlib import pyplot as plt
import cv2
from skimage import transform as trans
from PIL import Image
import numpy as np
import os
import pandas as pd

# 指定使用第二个GPU，第一个GPU容易爆炸
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

imgProcessing=FacialImageProcessing(False)

current_path = os.path.curdir

def preprocess(img, bbox=None, landmark=None, **kwargs):
    """
    This function will be used to preprocess the input image.
    For the case of the model, the input image should be a square image with the size of 224 x 224. 
    This function can reshape the input image and make the picture qualified.
    Plus, it will align the face, more details can be found in https://github.com/av-savchenko/hsemotion

    Args:
        img (_type_): input image
        bbox (_type_, optional): _description_. Defaults to None.
        landmark (_type_, optional): _description_. Defaults to None.

    Returns:
        return a qualified image
    """
    M = None
    image_size = [224,224]
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==224:
        src[:,0] += 8.0
    src*=2
    if landmark is not None:
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        #dst=dst[:3]
        #src=src[:3]
        #print(dst.shape,src.shape,dst,src)
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
        #print(M)

    if M is None:
        if bbox is None: #use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
              det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin//2, 0)
        bb[1] = np.maximum(det[1]-margin//2, 0)
        bb[2] = np.minimum(det[2]+margin//2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin//2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
              ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret 
    else: #do align using landmark
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
        return warped


def get_numpy_all(video_path_prefix, video_name, log2_path):
    """
    Thsi function will be used to extract the features of the video.
    'all' means the function will extract all the features of the video and average the extracted features if the video contains more than one faces.

    Args:
        video_path_prefix (string): the prefix of the video path, attentin !!! prefix !!!
        video_name (string): the video's name or the video's absolute path
        log2_path (string): log2 path which will be used for the number of extracted frames

    Returns:
        if the video is processable, the function will return "OK" and video's features with the shape of (1, 1280)
        if not, the function will return None and -1
    """
    # 如果当前的视频文件名字里面没有后缀名，我们自动加上
    if '.' not in video_name:
        video_name = video_name + ".mp4"
        
    # 如果输入的video_name是一个路径，我们自动提取文件名
    video_name = os.path.basename(video_name)
        
    video_name_without_extension = video_name.split(".")[0]
    
    video_path = video_path_prefix + "/" + video_name
    
    my_video = cv2.VideoCapture(video_path)
    
    if my_video.isOpened() is False:
        print(video_name, "this video can't be opened!")
        return None, -1

    # 设置my_video读取第一帧
    my_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frames_count = int(my_video.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("total frames: ", frames_count)

    features = []
    
    count = 0
    while True:
        ret, frame = my_video.read()
        
        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = np.asarray(frame)

        hse = facial_emotions.HSEmotionRecognizer()

        boxes, _ = imgProcessing.detect_faces(frame)
        
        # print("*******", boxes)

        bounding_boxes, points = imgProcessing.detect_faces(frame)
        points = points.T
        
        tmp_features = []
        for bbox,p in zip(bounding_boxes, points):
            box = bbox.astype(int)
            x1,y1,x2,y2=box[0:4]    
            p = p.reshape((2,5)).T
            
            face_img=preprocess(frame,box,None) #p)
            
            face_img=preprocess(frame,box,p)
            
            try:
                feature = hse.extract_features(face_img)
            except Exception as e:
                print(video_name, " : Can't extract the feature, maybe there do not have a face in the picture!")
                continue
                
            tmp_features.append(feature)


        if tmp_features == []:
            continue

        count += 1
        
        tmp_features = np.mean(tmp_features, axis=0)
        
        features.append(tmp_features)


        # 清理图像数据以释放显存
        del frame, boxes, bounding_boxes, points, tmp_features, face_img
        plt.close('all')
        hse = None
        
    if features == []:
        # print("this video can't be used!!!!!!!!!!!!")
        return None, None
    features = np.array(features)
    features = np.vstack(features)  # (extracted_frames_number, 1280)
    
    
    # 记录提取了多少帧
    df = pd.read_csv(log2_path)
    new_data = {
        'file': [video_name],
        'frames': [count],
    }
    new_data = pd.DataFrame(new_data)
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(log2_path, index=False)   
    
    # 第一维度平均
    features = np.mean(features, axis=0)
    
    my_video.release()
    cv2.destroyAllWindows()

    import gc
    gc.collect()

    return "OK", features


def get_numpy_all_whole(video_path_prefix, video_name, log2_path):
    """
    this function is used to process the unqulified video which means the video doesn't have the detectable face through the whole video.
    For the sake of simplicity, this function is copied from the get_numpy_all function and modified a little bit.
    This funciton select the whole picture to extract the feature instead of the face-detected area. 
    this approch is mainly served as two purposes:
        1. Keep the number of dialogues extracted from the scene conversation unchanged, that is, the first dimension remains unchanged
        2. Compare with the detectable picture

    Args:
        video_path_prefix (string): the prefix of the video path, attentin !!! prefix !!!
        video_name (string): the video's name or the video's absolute path
        log2_path (string): log2 path which will be used for the number of extracted frames
        
    return:
        the function will return "OK" and video's features with the shape of (1, 1280)
    """
    
     # 如果当前的视频文件名字里面没有后缀名，我们自动加上
    if '.' not in video_name:
        video_name = video_name + ".mp4"
        
    # 如果输入的video_name是一个路径，我们自动提取文件名
    video_name = os.path.basename(video_name)
    
    video_path = video_path_prefix + "/" + video_name
    
    my_video = cv2.VideoCapture(video_path)

    # 设置my_video读取第一帧
    my_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # frames_count = int(my_video.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("total frames: ", frames_count)

    features = []
    
    count = 0
    
    
    while True:
        ret, frame = my_video.read()
        
        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = np.asarray(frame)

        hse = facial_emotions.HSEmotionRecognizer()
        
        tmp_features = []
        feature = hse.extract_features(frame)
        # print("************")
        # print(feature)
        tmp_features.append(feature)


        if tmp_features == []:
            continue

        count += 1
        
        tmp_features = np.mean(tmp_features, axis=0)
        
        features.append(tmp_features)


        # 清理图像数据以释放显存
        del frame, tmp_features
        plt.close('all')
        hse = None
        
    if features == []:
        # print("this video can't be used!!!!!!!!!!!!")
        return None, None
    features = np.array(features)
    features = np.vstack(features)  # (extracted_frames_number, 1280)
    
    
    # 记录提取了多少帧
    df = pd.read_csv(log2_path)
    new_data = {
        'file': [video_name],
        'frames': [count],
    }
    new_data = pd.DataFrame(new_data)
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(log2_path, index=False)   
    
    # 第一维度平均
    features = np.mean(features, axis=0)
    
    my_video.release()
    cv2.destroyAllWindows()

    import gc
    gc.collect()

    return "OK", features
