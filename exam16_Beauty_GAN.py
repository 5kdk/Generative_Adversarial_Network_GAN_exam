import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf 
# import tensorflow._api.v2 as tf # 2.x 버전용 코드
tf.disable_v2_behavior() # tensorflow 2.x 버젼에서 1.x버젼 코드가 실행되도록
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')

img = dlib.load_rgb_image(('./imgs/12.jpg'))
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()