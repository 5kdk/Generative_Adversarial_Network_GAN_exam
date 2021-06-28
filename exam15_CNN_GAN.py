import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

OUT_DIR = './CNN_OUT_img/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# 변수 선언
img_shape = (28, 28, 1)
epoch = 10000
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

# build generator
generator_model = Sequential()
generator_model.add(Dense(256*7*7, input_dim=noise))
generator_model.add(Reshape((7, 7, 256)))
# Conv2DTranspose - 업 샘플링 이후 컨블루션
generator_model.add(Conv2DTranspose(128, kernel_size=3,
                                    strides=2, padding='same'))
# kernel_size:(3,3), stride=2 (2배로 업 샘플링)
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(Conv2DTranspose(64, kernel_size=3,
                                    strides=1, padding='same'))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))

#이진분류기 마지막 레이어 1로 출력
#strides: 커널을 씌울 떄 몇칸씩 이동하면서 씌울것인지, 실행하면 패딩을 줘도 사이즈가 줄어듬
generator_model.add(Conv2DTranspose(1, kernel_size=3,
                                    strides=2, padding='same'))
generator_model.add(Activation('tanh'))

generator_model.summary()



# build discriminator
discriminator_model = Sequential()
discriminator_model.add(Conv2D(32, kernel_size=3,
                               strides=2, padding='same', input_shape=img_shape))
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Conv2D(64, kernel_size=3,
                               strides=2, padding='same'))
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Conv2D(128, kernel_size=3,
                               strides=2, padding='same'))
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Flatten())
discriminator_model.add(Dense(1, activation='sigmoid'))
discriminator_model.summary()

discriminator_model.compile(loss='binary_crossentropy',
                            optimizer='adam', metrics=['acc'])
discriminator_model.trainable = False


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
            
