### Stacked autoencoder class
#
## 사용법
# ae = autoencoder() # 객체 생성
# ae.make(x)         # x 데이터 셋에 맞춰 그래프 만들기
# ae.train(x)        # 학습하기
#
## ae.train(x) 리턴값
# encoding, outputs = ae.train(x)
# encoding: 중요한 feature 로 인코딩된 정보들
# outputs: encoding 정보를 바탕으로 디코딩한 출력들
#
## 그 외 메소드들
# set_layer(hidden1, hidden2) : 히든 레이어 사이즈 설정
# set_train(n_epochs, learning_rate, batch_size, l2_reg=0.0001) : 학습 파라미터 설정
#
## 멤버 변수
# shape0 : 전체 데이터 수
# n_inputs : 데이터 feature 수 (df.shape[1])
# n_hidden1 : 인코더 레이어 크기
# n_hidden2 : 인코딩 유닛 크기
# n_hidden3 : 디코더 레이어 크기 (=n_hidden1)
# n_outputs : 디코딩된 출력 크기 (=n_inputs)
# n_epochs : 총 에폭 수
# learning_rate : 학습률
# batch_size : 미니배치 크기
# l2_reg : l2 규제율 (https://kolikim.tistory.com/50) (https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow)


import tensorflow as tf
import numpy as np
import pandas as pd
from functools import partial
from datetime import datetime
import os


class Autoencoder():
    # 생성자. 사용될 변수 미리 선언
    def __init__(self, n_hidden1=40, n_hidden2=20, n_hidden3=10):
        self.shape0 = None               # 전체 데이터 수
        
        self.n_inputs = None             # input size
        self.n_hidden1 = n_hidden1       # encoder layer1 size
        self.n_hidden2 = n_hidden2       # encoder layer2 size
        self.n_hidden3 = n_hidden3       # coding units size (like PCA)
        self.n_hidden4 = self.n_hidden2  # decoder layer2 size (= encoder layer2 size)
        self.n_hidden5 = self.n_hidden1  # decoder layer1 size (= encoder layer1 size)
        self.n_outputs = self.n_inputs   # restruction
        
        self.n_epochs = 3
        self.learning_rate = 0.01
        self.batch_size = 150
        self.l2_reg = 0.0001

    def parameters(self):
        print(f'shape0 : {self.shape0}')
        print(f'n_inputs : {self.n_inputs}')
        print(f'n_hidden1 : {self.n_hidden1}')
        print(f'n_hidden2 : {self.n_hidden2}')
        print(f'n_hidden3 : {self.n_hidden3}')
        print(f'n_hidden4 : {self.n_hidden4}')
        print(f'n_hidden5 : {self.n_hidden5}')
        print(f'n_outputs : {self.n_outputs}')
        print()
        print(f'n_epochs : {self.n_epochs}')
        print(f'learning_rate : {self.learning_rate}')
        print(f'batch_size : {self.batch_size}')
        print(f'n_batches : {self.n_batches}')
        print(f'l2_reg : {self.l2_reg}')
        print()
    
    # 레이어 사이즈 설정
    def set_layer(self, n_hidden1, n_hidden2, n_hidden3):
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.n_hidden4 = self.n_hidden2
        self.n_hidden5 = self.n_hidden1
        self.n_outputs = self.n_inputs

    # 학습 파라미터 설정
    def set_train(self, n_epochs, learning_rate, batch_size, l2_reg=0.0001):
        if not self.shape0:
            print('메소드 make_autoencoder 를 먼저 실행해주세요')
            return
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_batches = self.shape0 // batch_size
        self.l2_reg = l2_reg
        
    def make(self, df):
        self.shape0 = df.shape[0]
        self.n_inputs = df.shape[1]
        self.n_outputs = self.n_inputs # autoencoder: input_size==output_size
        if self.batch_size > df.shape[0]:
            print(f'batch_size가 {self.shape0}로 변경됨')
            self.batch_size = self.shape0
        self.n_batches = self.shape0 // self.batch_size
        
        he_init = tf.keras.initializers.he_normal() # He 초기화
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg) # L2 규제
        
        # partial 을 이용한 tf.layers.dense 의 새 버전 만들기
        # partial 첫번째 인자는 새 버전 만들 원본.
        # 나머지 인자는 원본 함수가 갖는 인자를 재정의 하는데 쓰인다.
        dense_layer = partial(tf.layers.dense,
                              activation=tf.nn.relu,
                              kernel_initializer=he_init,
                              kernel_regularizer=l2_regularizer)
        
        # Stacked Automater 구성
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.n_inputs])

        self.hidden1 = dense_layer(self.inputs, self.n_hidden1)
        self.hidden2 = dense_layer(self.hidden1, self.n_hidden2)
        self.hidden3 = dense_layer(self.hidden2, self.n_hidden3)
        self.hidden4 = dense_layer(self.hidden3, self.n_hidden4)
        self.hidden5 = dense_layer(self.hidden4, self.n_hidden5)
        self.coding_units = self.hidden3

        self.outputs = dense_layer(self.hidden5, self.n_outputs, activation=None)

        # loss
        self.reconstruction_loss = tf.reduce_mean(tf.square(self.outputs - self.inputs)) # mse
        self.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([self.reconstruction_loss] + self.reg_losses)

        # optimizer
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

    def shuffle_batch(self, features, seed):
#         np.random.seed(seed) # 시드 설정시 학습이 안됨. 주석처리.
        shuffled_index = np.random.permutation(self.n_inputs)
        for batch_idx in np.array_split(shuffled_index, self.n_batches):
            batch_x = features[batch_idx]
        yield batch_x
        
    def train(self, train_x, tag=''):
        print(':::::::: Autoencoding training start')
        if type(train_x) is not np.ndarray:
            train_x = np.asarray(train_x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.n_epochs):
                for _ in range(self.n_batches):
                    batch_x = next(self.shuffle_batch(train_x, epoch+datetime.now().hour)) # seed: 현재 시간+epoch
                    _, _loss = sess.run([self.train_op, self.reconstruction_loss], feed_dict={self.inputs:batch_x})
                print(f'epoch: {epoch+1}/{self.n_epochs}, Train MSE: {_loss:.10f}')
            print()
            coding_units = sess.run(self.coding_units, feed_dict={self.inputs:train_x})
            outputs = sess.run(self.outputs, feed_dict={self.inputs:train_x})
            # 학습 끝나면 레이어 중앙의 인코딩 유닛과 디코딩된 아웃풋 리턴
            return coding_units, outputs 
        
            # train/ 디렉토리에 모델 저장합니다.
            # self.saver.save(sess, f'trained/trained_autoencoder_{tag}')

#     # 저장된 모델 불러오기 및 인코딩유닛 리턴하기 --미구현--
#     # 'Attempting to use uninitialized value' error..
#     def get(self, tag=''):
#         with tf.Session() as sess:
#             if os.path.exists(f'trained/trained_autoencoder_{tag}'):
#                 saver = tf.train.import_meta_graph(f'trained/trained_autoencoder_{tag}.meta')
#                 saver.restore(sess, f'trained/trained_autoencoder_{tag}')
#             coding_units_val = sess.run(self.coding_units, feed_dict={self.inputs:df_small[:2]})
#         return coding_units
