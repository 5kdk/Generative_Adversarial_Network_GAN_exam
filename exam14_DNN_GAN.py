import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

# 변수 선언
OUT_DIR = './DNN_OUT_img/'
img_shape = (28, 28, 1)
epoch = 100000
batch_size = 128
noise = 100
sample_interval = 100

# 데이터 불러오기
# train 데이터만 불러오기
(X_train , _), (_, _) = mnist.load_data()
print(X_train.shape)

X_train = X_train / 127.5 - 1 # -1 ~ 1 사이의 값을 가지도록 스케일링
X_train = np.expand_dims(X_train, axis = 3) # 차원추가 (reshape와 같은 역할 수행)
print(X_train.shape)


# build generator (생성자)
generator_model = Sequential()
generator_model.add(Dense(128, input_dim=noise)) # 100개 짜리 잡음으로 제공
generator_model.add(LeakyReLU(alpha=0.01)) # LeakyReLU (activation function) /  알파 값을 주기위해 따로 작성 
# 값에 음수가 있기 때문에 relu 대신 사용(음수에도 약간의 기울기 가짐) / activation function(layer x)
generator_model.add(Dense(784, activation='tanh'))
generator_model.add(Reshape(img_shape))
print(generator_model.summary())


# build discriminator (구분자)
# lrelu = LeakyReLU(alpha=0.01)
discriminator_model = Sequential()
discriminator_model.add(Flatten(input_shape=img_shape)) # 원본데이터 손실없이(reshape 없이)사용 / 모델에 Flatten을 추가
# discriminator_model.add(Dense(128, activation='lrelu'))  # lrelu 변수 선언 후 한줄로도 처리도 가능
discriminator_model.add(Dense(128))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Dense(1, activation='sigmoid')) # 참 거짓 이진분류 - sigmoid 사용
print(discriminator_model.summary())


discriminator_model.compile(loss='binary_crossentropy',
                            optimizer='adam', metrics=['acc'])
discriminator_model.trainable = False # 학습 x(backward x), 생성자와 구분자의 동등한 대립을 위해 학습제어


# build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
gan_model.compile(loss='binary_crossentropy', optimizer='adam')
print(gan_model.summary())

# real, fake label 생성
real = np.ones((batch_size, 1)) # 모든 값이 1인 행렬
print(real)
fake = np.zeros((batch_size, 1)) # 모든 값이 0인 행렬
print(fake)

for itr in range(epoch):
    idx = np.random.randint(0, X_train.shape[0], batch_size) # 0 ~ 60000까지 128개
    real_imgs = X_train[idx] # 랜덤한 128개 이미지
    
    z = np.random.normal(size=(batch_size, noise)) # 정규분포 노이즈 (128, 100)
    fake_imgs = generator_model.predict(z) # generator가 noise로 생성한 이미지
    
    # batch size만큼 한번 학습
    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)
    
    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake) # real과 fake의 평균 loss, accuracy
    
    
    z = np.random.normal(size=(batch_size, noise))  # 정규분포 노이즈 (128, 100)
    gan_hist = gan_model.train_on_batch(z, real) # fake img에 label은 real로
    
    if itr % sample_interval == 0: # 100번마다 출력
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]'%(
            itr, d_loss, d_acc*100, gan_hist))
        row = col = 4
        z = np.random.normal(size=(row*col, noise)) # (16, 100)
        fake_imgs = generator_model.predict((z))
        fake_imgs = 0.5 * fake_imgs + 0.5 # 0 ~ 1 사이의 값을 가지도록 스케일링
        _, axs = plt.subplots(row, col, figsize=(row, col), sharey=True, sharex=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(itr+1))
        plt.savefig(path)
        plt.close()
        
        