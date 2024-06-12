from hsemotion import facial_emotions
from facial_analysis import FacialImageProcessing
from matplotlib import pyplot as plt
import cv2
from skimage import transform as trans

imgProcessing=FacialImageProcessing(False)

from PIL import Image

import numpy as np


def preprocess(img, bbox=None, landmark=None, **kwargs):
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

# 这里修改为测试的图片的路径
im = Image.open("/Users/zhangqi/Desktop/HSE/test_img/frame_7.jpg")


print("this is im:", im)
frame = np.asarray(im)
print("this is a:", frame)

hse = facial_emotions.HSEmotionRecognizer()

try:
    feature = hse.extract_features(frame)
except Exception as e:
    
    print("Can't extract the feature, maybe there do not have a face in the picture!")
    exit(0)

print("this is feature:", feature)
print("feature shape", feature.shape)



bounding_boxes, points = imgProcessing.detect_faces(frame)
points = points.T



for bbox,p in zip(bounding_boxes, points):
    box = bbox.astype(int)
    x1,y1,x2,y2=box[0:4]    
    p = p.reshape((2,5)).T
        
    # plt.figure(figsize=(5, 5))
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    face_img=preprocess(frame,box,None)
    # ax1.set_title('Cropped')
    # ax1.imshow(face_img)
    
    face_img=preprocess(frame,box,p)
    # ax2.set_title('Aligned')
    # ax2.imshow(face_img)
    
    feature = hse.extract_features(face_img)

    # plt.show()
    # print("this is feature:", feature)
    # print("this is feature shape:", feature.shape)