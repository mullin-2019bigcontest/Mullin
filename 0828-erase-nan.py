import pandas as pd
import numpy as np

# 28 day로 채워진 csv 불러옵니다.
data = pd.read_csv('data/day_filled_train.csv')

df = data.copy()

## 변수명 알아보기 쉽게 정리
# 0828-erase-nan.ipynb 주석 참고
df.rename(columns={'class_0':'c_cnt_class0', # c_cnt_class0 : "combat count class0", 해당 날짜에 class0 으로 전투한 횟수
                  'class_1':'c_cnt_class1',
                  'class_2':'c_cnt_class2',
                  'class_3':'c_cnt_class3',
                  'class_4':'c_cnt_class4',
                  'class_5':'c_cnt_class5',
                  'class_6':'c_cnt_class6',
                  'class_7':'c_cnt_class7',
                  'etc_cnt':'c_etc_cnt', 
                  'num_opponent':'c_num_opponent',
                  'pledge_cnt':'c_pledge_cnt',
                  'random_attacker_cnt':'c_random_attacker_cnt',
                  'random_defender_cnt':'c_random_defender_cnt',
                  'same_pledge_cnt':'c_same_pledge_cnt',
                  'temp_cnt':'c_temp_cnt',
                  'avg_play_rate_per_pledge':'avg_play_rate_rank_per_p',
                  'combat_char_cnt':'p_c_char_cnt', # pledge 필드, combat char count
                  'combat_play_time':'p_c_sum_play_time', # pledge 필드, combat sum play time
                  'non_combat_play_time':'p_non_c_sum_play_time',
                  'play_char_cnt':'p_play_char_cnt',
                  'pledge_combat_cnt':'p_c_cnt',
                  'total_combat_cnt_per_pledge':'tot_c_rank_per_p',
                  'total_item_price':'tot_item_price',
                  'time_bin_0':'t_time_bin_0',
                  'time_bin_1':'t_time_bin_1',
                  'time_bin_2':'t_time_bin_2',
                  'time_bin_3':'t_time_bin_3',
                  'type_0':'t_type_0',
                  'type_1':'t_type_1',
                  'death':'a_death_cnt',
                  'enchant_count':'a_enchant_count',
                  'exp_recovery':'a_exp_recovery_cnt',
                  'fishing':'a_fish',
                  'game_money_change':'a_money_change',
                  'npc_kill':'a_npc_kill',
                  'party_exp':'a_party_exp',
                  'playtime':'a_playtime',
                  'private_shop':'a_private_shop',
                  'quest_exp':'a_quest_exp',
                  'revive':'a_revive_cnt',
                  'rich_monster':'a_boss_monster',
                  'solo_exp':'a_solo_exp',
                  'total_exp':'tot_exp',
                  }, inplace=True)

# char_id drop
df.drop('char_id', axis=1, inplace=True)


## 0으로 치환할 컬럼, 평균으로 치환할 컬럼 구분
cols_list = df.columns.values.tolist()
cols_list.remove('acc_id')
cols_list.remove('day')

cols_to_mean = ['p_c_char_cnt', 'p_c_sum_play_time', 'p_non_c_sum_play_time', 'p_etc_cnt',
				'p_play_char_cnt', 'p_c_cnt', 'p_random_attacker_cnt', 'p_random_defender_cnt',
				'p_same_pledge_cnt', 'p_temp_cnt', 'tot_c_rank_per_p', 'isMajorClass',
				'avg_play_rate_rank_per_p']

for col in cols_to_mean:
	cols_list.remove(col)

cols_to_0 = cols_list

## NaN -> 0 으로 치환
df[cols_to_0] = df[cols_to_0].fillna(0)


## NaN -> 각 acc_id 가 갖는 value 평균으로 치환
for col in cols_to_mean:
	df[col] = df.groupby('acc_id')[col].transform(lambda x: x.fillna(x.mean()))
# 평균조차 NaN 이었던 NaN행, 0 으로 치환	
df[cols_to_mean] = df[cols_to_mean].fillna(0)



## 파일 저장
df.to_csv('data/28day_no_nan_train.csv')