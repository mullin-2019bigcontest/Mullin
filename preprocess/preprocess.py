import pandas as pd
import os
from datetime import datetime

# ------------------------------------------------------------------------------------
#      variable definition
# ------------------------------------------------------------------------------------
fields = ['activity', 'combat', 'pledge', 'trade', 'payment']

# 경로
current_working_dir = os.getcwd()
root_dir = os.path.dirname(current_working_dir)
raw_dir = os.path.join(root_dir, 'raw')
preprocess_dir = os.path.join(root_dir, 'preprocess')


# ------------------------------------------------------------------------------------
#      function definition
# ------------------------------------------------------------------------------------
from module.base_preprocessing import *
from module.replace_nan import *

def preprocess(dataset):
	dfs = load_data(dataset, raw_dir)         # raw csv 로드
	df = merge_df(dataset, dfs)               # 한개 csv로 merge
	df = fill_day(dataset, df)                # 28 days 채우기 (오랜시간 소요)
	df = replace_nan(dataset, df)
	save_df(dataset, df, preprocess_dir)      # 저장

def load_data(dataset, raw_dir):
	print('\n:::::::: create dataframes...')
	dfs = list()
	for field in fields:
		name = f'{dataset}_{field}'
		path = os.path.join(raw_dir, f'{name}.csv')
		dfs.append(pd.read_csv(path))
		print(name, '\t', dfs[-1].shape)
	return dfs

def merge_df(dataset, dfs):
	print('\n:::::::: prepare to merge...')
	acc_id = pd.DataFrame(dfs[1]['acc_id'].unique(), columns=['acc_id'])
	activity = preprocessing_activity(dfs[0])
	combat = preprocessing_combat(dfs[1])
	pledge = preprocessing_pledge(dfs[2])
	trade = preprocessing_trade(dfs[3], acc_id) # trade 는 기준 acc_id 필요
	payment = preprocessing_payment(dfs[4])

	# Squeeze the whole dataframes into one
	print('\n:::::::: merge dataframes...')
	df = combat.join(pledge).join(trade).join(activity)
	print(f'{dataset} shape:\t{df.shape}')
	return df

def fill_day(dataset, df):
	print('\n:::::::: fill 28 days each acc_id...')
	df.reset_index(level=['acc_id', 'day'], inplace=True) # multi-index 해제
	blank = pd.DataFrame()
	length = len(df.acc_id.unique())
	print(f'progress: {0:5d}/{length:5d}')
	for i, _id in enumerate(df.acc_id.unique()):
		if (i+1) % 5000 == 0: print(f'progress: {i+1:5d}/{length:5d}')
		temp = pd.DataFrame({'acc_id':_id, 'day':range(1,29)}) # acc_id 하나에 대해 1~28까지 day를 갖는 28개 row를 갖는 데이터프레임 1개 만들기 (acc_id 갯수만큼 반복)
		blank = pd.concat([blank,temp], axis=0)  # 만든 데이터프레임을 빈 데이터프레임에 행방향으로 붙이기
	# blank : 4만개 acc_id 가 28개씩 day를 갖는 데이터프레임
	day_filled_df = blank.merge(df, how='outer', on=['acc_id', 'day'])
	print(f'{dataset} shape:\t{day_filled_df.shape}')
	return day_filled_df

def save_df(dataset, df, preprocess_dir):
	print('\n:::::::: save dataframe to csv...')
	time = datetime.now().strftime('%m%d-%H')
	name = f'{time}-{dataset}.csv'
	path = os.path.join(preprocess_dir, name)

	df.to_csv(path, index=False)
	print(f'\nDone. {path}\n')
	


# ------------------------------------------------------------------------------------
#      main
# ------------------------------------------------------------------------------------

preprocess('train')

# preprocess('test1')

# preprocess('test2')
