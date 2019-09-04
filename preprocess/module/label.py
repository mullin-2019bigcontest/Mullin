import pandas as pd
import os
from .path_header import *

LABEL_PATH = os.path.join(RAW_DIR, 'train_label.csv')

# preprocess/train_label_ohe.csv 로 저장합니다.
# acc_id 기준으로 정렬 + one-hot-encoding 수행
def ohe():
	label = pd.read_csv(LABEL_PATH)

	# train 데이터가 acc_id 기준으로 오름차순 정렬이기 때문에 label 데이터도 동일하게 맞춥니다.
	label = label.sort_values('acc_id', ascending=True).reset_index(drop=True)
	# survival_time -> one-hot-encoding 하기
	label = pd.concat((label['acc_id'], pd.get_dummies(label['survival_time']), label['amount_spent']), axis=1)

	file_path = os.path.join(PREPROCESS_DIR, 'train_label_ohe.csv')
	label.to_csv(file_path, index=False)

# preprocess/train_label.csv 로 저장합니다.
# acc_id 기준 정렬만 수행
def no_ohe():
	label = pd.read_csv(LABEL_PATH)

	# train 데이터가 acc_id 기준으로 오름차순 정렬이기 때문에 label 데이터도 동일하게 맞춥니다.
	label = label.sort_values('acc_id', ascending=True).reset_index(drop=True)

	file_path = os.path.join(PREPROCESS_DIR, 'train_label.csv')
	label.to_csv(file_path, index=False)





