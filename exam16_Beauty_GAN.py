import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import tensorflow._api.v2 as tf # 2.x 버전용 코드
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior() # tensorflow 2.x 버젼에서 1.x버젼 코드가 실행되도록
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')
    
img = dlib.load_rgb_image(('./imgs/12.jpg'))
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()

# face_detector으로 얼굴 인식
img_result = img.copy()
dets = detector(img) # 얼굴의 좌표
if len(dets) == 0:
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1, figsize=(16, 10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y),w,h, linewidth=2, edgecolor='r',
                                 facecolor='none') 
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()
    

# 모델을 활용해서 인식    
fig,  ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3,
                                edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
plt.show()

faces = dlib.get_face_chips(img,objs, size=256, padding=0.3)
fig, axes = plt.subplot(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)
plt.show()
