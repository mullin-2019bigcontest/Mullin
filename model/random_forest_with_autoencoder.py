# Mullin/model/random_forest.py

from module.path_header import *  # 경로 정리해둔 헤더 파일
from module.autoencoder import Autoencoder
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import pickle
import time

TRAIN_NAME = '0831-21-train.csv'
LABEL_NAME = 'train_label.csv'
TEST1_NAME = '0831-21-test1.csv'
TEST2_NAME = '0831-21-test2.csv'
MODEL1_NAME = 'random_forest_ae_model1.pkl' # random_forest_autoencoder_model1 (survival_time)
MODEL2_NAME = 'random_forest_ae_model2.pkl' # random_forest_autoencoder_model2 (amount_spent)

TRAIN_PATH = os.path.join(PREPROCESS_DIR, TRAIN_NAME) 
LABEL_PATH = os.path.join(PREPROCESS_DIR, LABEL_NAME)
TEST1_PATH = os.path.join(PREPROCESS_DIR, TEST1_NAME)
TEST2_PATH = os.path.join(PREPROCESS_DIR, TEST2_NAME)
MODEL1_PATH = os.path.join(MODEL_DIR, MODEL1_NAME)  # survival_time prediction model
MODEL2_PATH = os.path.join(MODEL_DIR, MODEL2_NAME)  # amount_spent prediction model

## main function
# survival_time, amount_spent 에 대한 모델 각각 만들고 model/ 에 저장한다.
# size=40000 (전체 train dataset) 으로 하면 시간 오래걸린다.
def create_model_rf(train_X, train_y, size=1000):    
    # train_test_split
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, train_size=0.8)     
    
    print('create survival time model')
    survival_time_model(size, train_X, val_X, train_y, val_y)
    print('create amount spent model')
    amount_spent_model(size, train_X, val_X, train_y, val_y)
    

# random_forest 에 맞게 train input 형태 조정
# 1~28 day 무시하고 acc_id 에 대한 값으로 squeeze
def preprocess_X(train_X):
    ## 합할 feature, 평균낼 feature 나누기
    # to mean features
    mean_features = ['isMajorClass', 'avg_play_rate_rank_per_p', 'tot_c_rank_per_p']
    # to sum features
    sum_features = train_X.columns.tolist()[2:]
    for feat in mean_features:
        sum_features.remove(feat)
    # acc_id 에 대해 mean_features, sum_features 컬럼 평균/합 한 value들 concat 하기
    # 의미: 1~28day 무시하고 feature들을 acc_id 에 대한 값으로 squeeze 하기
    mean_pivot = train_X.pivot_table(index='acc_id', values=mean_features, aggfunc='mean')
    sum_pivot = train_X.pivot_table(index='acc_id', values=sum_features, aggfunc='sum')
    train_X = pd.concat((mean_pivot, sum_pivot), axis=1)        

    # reset_index + acc_id 컬럼 지우기 (acc_id 인덱스, 따로 순서대로 저장해서 필요x)
    train_X = train_X.reset_index(drop=True)

    return train_X

def preprocess_y_survival(train_y):
    train_y = train_y.iloc[:,1]  # survival_time column 추출
    # random_forest label 에 넣기 위해 스칼라 값으로 reshape
    train_y = train_y.values.reshape(-1,) # reshape (40000, )
    return train_y

def preprocess_y_spent(train_y):
    train_y = train_y.iloc[:,-1]  # amount_spent column 추출
    # random_forest label 에 넣기 위해 스칼라 값으로 reshape
    train_y = train_y.values.reshape(-1,) # reshape (40000, )
    return train_y



def survival_time_model(size, train_X, val_X, train_y, val_y):
    train_y = preprocess_y_survival(train_y)
    val_y = preprocess_y_survival(val_y)

    # rfc = RandomForestClassifier(n_estimators=50, max_depth=70, n_jobs=-1, random_state=42)
    # rfc.fit(train_X[:size], train_y[:size])

    # score = rfc.score(val_X, val_y)
    # predict = rfc.predict(val_X)

    from sklearn.model_selection import GridSearchCV
    random_classifier = RandomForestClassifier(n_jobs=-1)
    params = {'n_estimators':[50,100], 'max_features':[None], 'max_depth':[5,10], 'max_leaf_nodes':[50,100]}
    rfc = GridSearchCV(random_classifier, n_jobs=-1, param_grid=params, cv=5)

    rfc.fit(train_X[:size], train_y[:size])
    score = rfc.score(val_X, val_y)
    predict = rfc.predict(val_X)

    print(rfc.best_params_)
    
    print(f'validation dataset 에 대한 score: {score:.4f}')
    print(f'validation dataset 의 분류된 label 수: {len(np.unique(predict))}')
    print()
    
    # 모델 저장 (model/*_model1.pkl)
    with open(MODEL1_PATH, 'wb') as fp:
        pickle.dump(rfc, fp)
        
def amount_spent_model(size, train_X, val_X, train_y, val_y):
    train_y = preprocess_y_spent(train_y)
    val_y = preprocess_y_spent(val_y)

    rfr = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    rfr.fit(train_X[:size], train_y[:size])

    score = rfr.score(val_X, val_y)
    predict = rfr.predict(val_X)
    mse = mean_squared_error(val_y, predict)
    
    print(f'validation dataset 에 대한 mse score: {mse:.4f}')
    print()
    
    # 모델 저장 (model/*_model2.pkl)
    with open(MODEL2_PATH, 'wb') as fp:
        pickle.dump(rfr, fp)

# 저장된 모델 불러와서 test dataset 에 대해 예측
def test_model_rf(train, test1, test2):  
    train_y1 = preprocess_y_survival(pd.read_csv(LABEL_PATH))
    train_y2 = preprocess_y_spent(pd.read_csv(LABEL_PATH))

    with open(MODEL1_PATH, 'rb') as fp:
        model = pickle.load(fp)
        score = model.score(train, train_y1)
        predict1 = model.predict(test1)
        predict2 = model.predict(test2)
        
        print('survival time')
        print(f'전체 train dataset 에 대한 score: {score:.4f}')
        print(f'분류된 test1 dataset 의 label 수: {len(np.unique(predict1))}')
        print(f'분류된 test2 dataset 의 label 수: {len(np.unique(predict2))}')
        print()
        
    with open(MODEL2_PATH, 'rb') as fp:
        model = pickle.load(fp)
        predict0 = model.predict(train)
        predict1 = model.predict(test1)
        predict2 = model.predict(test2)

        mse = mean_squared_error(train_y2, predict0)

        print('amount spent')
        print(f'전체 train dataset 에 대한 mse: {mse:.4f}')
        print(f'predict1: {predict1}')
        print(f'{len(np.unique(predict1))}')
        print()


# ------------------
#   main
# ------------------
start = time.time()  # 코드 시작 시간

train_X = pd.read_csv(TRAIN_PATH)
train_y = pd.read_csv(LABEL_PATH)
test1 = pd.read_csv(TEST1_PATH)
test2 = pd.read_csv(TEST2_PATH)

train_X = preprocess_X(train_X)
test1 = preprocess_X(test1)
test2 = preprocess_X(test2)

# scaling
# 전체 train 데이터셋에 대해 fit_transform
# test 데이터셋에 대해 transform
mm = MinMaxScaler()
train_X = mm.fit_transform(train_X)
test1 = mm.transform(test1)
test2 = mm.transform(test2)

## Autoencoding
ae = Autoencoder()
# train
ae.make(train_X)
ae.set_train(n_epochs=15, learning_rate=0.05, batch_size=10000, l2_reg=0.0001)
ae.set_layer(n_hidden1=40, n_hidden2=30, n_hidden3=20)
ae.parameters()
train_X, outputs = ae.train(train_X)
# test1
test1, outputs = ae.train(test1)
# test2
test2, outputs = ae.train(test2)

create_model_rf(train_X, train_y, size=40000)
test_model_rf(train_X, test1, test2)

exe_time = time.time() - start
print(f'execution time : {exe_time//3600:02.0f}h {exe_time%3600//60:02.0f}m {exe_time%60:02.0f}s')