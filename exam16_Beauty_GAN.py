import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import tensorflow._api.v2 as tf # 2.x 버전용 코드
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior() # tensorflow 2.x 버젼에서 1.x버젼 코드가 실행되도록
import numpy as np

# 변수 선언, 모델 불러오기
detector = dlib.get_frontal_face_detector() #얼굴 영역 인식 모델 로드
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')
    
# img = dlib.load_rgb_image(('./imgs/12.jpg'))
# plt.figure(figsize=(16,10))
# plt.imshow(img)
# plt.show()


# # face_detector으로 얼굴 인식
# img_result = img.copy()
# dets = detector(img) # 얼굴의 좌표
# if len(dets) == 0: #얼굴영역의 갯수가 0일 경우
#     print('cannot find faces!')
# else:
#     fig, ax = plt.subplots(1, figsize=(16, 10))
#     for det in dets:
#         x, y, w, h = det.left(), det.top(), det.width(), det.height()
#         rect = patches.Rectangle((x,y),w,h, linewidth=2, edgecolor='r',
#                                  facecolor='none') 
#         ax.add_patch(rect)
#     ax.imshow(img_result)
#     plt.show()
    

# # 모델을 활용해서 인식    
# fig, ax = plt.subplots(1, figsize=(16,10))
# objs = dlib.full_object_detections() #얼굴 수평맞춰줄때 사용
# for detection in dets:
#     s = sp(img, detection) #sp() : 얼굴의 랜드마크를 찾기
#     objs.append(s)
#     for point in s.parts(): #5개의 점에 대한 for문
#         circle = patches.Circle((point.x, point.y), radius=2,
#                                 edgecolor='r', facecolor='r')
#         ax.add_patch(circle)
# ax.imshow(img_result)
# plt.show()

# # get_face_chips로 세부 인식
# faces = dlib.get_face_chips(img,objs, size=256, padding=0.3)
# fig, axes = plt.subplot(1, len(faces)+1, figsize=(20,16))
# axes[0].imshow(img)
# for i, face in enumerate(faces):
#     axes[i+1].imshow(face)
# plt.show()


#원본이미지를 넣으면 align 완료된 얼굴이미지 반환하는 함수
def align_faces(img, detector, sp): 
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
    return faces


# # 위 함수 테스트용 코드
# test_img = dlib.load_rgb_image("./imgs/02.jpg")
# test_faces = align_faces(test_img)
# fig, axes = plt.subplots(1, len(test_faces) + 1, figsize=(20, 16))
# axes[0].imshow(test_img)
# for i, face in enumerate(test_faces):
#     axes[i + 1].imshow(face)
# plt.show()

def preprocess(img):
    return (img / 255.0 - 0.5) * 2 # 스케일링 (0 ~ 255 -> -1 ~ 1)

def deprocess(img):
    return (img + 1) / 2 # 스케일링 (-1 ~ 1 -> 0 ~ 255)


# 모델 로드
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph("./models/model.meta")
saver.restore(sess, tf.train.latest_checkpoint("./models"))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0") # source
Y = graph.get_tensor_by_name("Y:0") # reference
Xs = graph.get_tensor_by_name("generator/xs:0") # output

img1 = dlib.load_rgb_image("./imgs/12.jpg") # 화장 안한 사진
img1_faces = align_faces(img1, detector, sp)

img2 = dlib.load_rgb_image("./imgs/makeup/vFG56.png") # 화장 한 사진
img2_faces = align_faces(img2, detector, sp)

fig, axes = plt.subplots(1, 2, figsize=(16, 10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()

src_img = img1_faces[0] # 소스 이미지 
ref_img = img2_faces[0] # 레퍼런스 이미지

X_img = preprocess(src_img) # 스케일링
X_img = np.expand_dims(X_img, axis=0) # np.expand_dims() : 배열에 차원을 추가(reshape), (256,256,2) -> (1,256,256,3)

Y_img = preprocess(ref_img) 
Y_img = np.expand_dims(Y_img, axis=0) # 텐서플로에서 0번 axis는 배치 방향

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})
output_img = deprocess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()