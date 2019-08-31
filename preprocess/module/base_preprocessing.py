import pandas as pd

def preprocessing_activity(activity):
	print('preprocessing activity table')
	activity.drop('char_id', axis=1, inplace=True)
	total_play = activity.groupby('acc_id')['playtime'].sum()
	total_play = pd.DataFrame({'total_play_time': total_play}) # merge 위해 생성

	activity = pd.pivot_table(data=activity.drop('server', axis=1), 
								index=['acc_id', 'day'], aggfunc=sum) # 하루동안 모든 value의 합
	activity["total_exp"] = activity["solo_exp"] + activity["party_exp"] + activity["quest_exp"] # 총 exp 획득량
	activity["fishing_prop"] = activity["fishing"] / activity["playtime"] # 하루에 낚시에 투자하는 비율
	return activity

def preprocessing_combat(combat):
	print('preprocessing combat table')
	combat.drop(['char_id', 'server'], axis=1, inplace=True)
	combat['class'] = combat['class'].astype('category') # 'class' feature 카테고리화
	prop_class = combat['class'].value_counts() / combat['class'].value_counts().sum() # 클래스별 전체 비율

	# 전체 누적합 60% 미만을 차지하는 주류 클래스 조사
	class60 = prop_class[~(prop_class.cumsum()>0.6)].index # 기사, 요정, 마법사 (70%로 하면 전사까지.)
	combat['isMajorClass'] = combat['class'].apply(lambda x: 1 if x in class60 else 0) 

	max_level = combat.groupby('acc_id')['level'].max() # 계정의 최대 레벨
	mean_level = combat.groupby('acc_id')['level'].mean() # 계정의 평균 레벨
	combat.drop('level', axis=1, inplace=True) # 기존 level feature drop

	combat = pd.get_dummies(combat) # class feature 에 대해 get_dummies 적용됨

	# 한 계정에 대해 주류 클래스만 플레이 했을경우 1.0 아니면 비주류 플레이한 만큼 지표(평균) 떨어짐
	combat = pd.concat([pd.pivot_table(data=combat.drop('isMajorClass', axis=1), index=["acc_id", "day"], aggfunc=sum), pd.pivot_table(data=combat, index=['acc_id', 'day'], values='isMajorClass', aggfunc='mean')], axis=1)
	return combat

def preprocessing_pledge(pledge):
	print('preprocessing pledge table')
	pledge_pivot = pd.pivot_table(data=pledge, index=['pledge_id'], values='play_char_cnt', aggfunc='mean') # 각 혈맹의 평균 유저 접속률
	# 접속률 1위 혈맹이 1.0 score 를 갖는 지표
	avg_play_rate_per_pledge = (pledge_pivot.play_char_cnt.sort_values(ascending=False)\
	                                    / pledge_pivot.play_char_cnt.sort_values(ascending=False).iloc[0])
	to_be_merged = pd.DataFrame({'avg_play_rate_per_pledge':avg_play_rate_per_pledge})
	pledge = pd.merge(pledge, to_be_merged, on='pledge_id')

	# 혈맹간 전투 컨텐츠를 가장 많이 즐긴 혈맹 순위 지표
	pledge_pivot = pd.pivot_table(data=pledge, index=['pledge_id'], values='pledge_combat_cnt', aggfunc='sum') # 혈맹간 총 전투 수 
	total_combat_cnt_per_pledge = (pledge_pivot.pledge_combat_cnt.sort_values(ascending=False)\
                         	       / pledge_pivot.pledge_combat_cnt.sort_values(ascending=False).iloc[0])
	to_be_merged = pd.DataFrame({'total_combat_cnt_per_pledge':total_combat_cnt_per_pledge})
	pledge = pd.merge(pledge, to_be_merged, on='pledge_id')

	# 각 혈맹의 유저 수 
	pledge_num_people = pledge["pledge_id"].value_counts()
	pledge.drop(["pledge_id"], axis=1, inplace=True)

	pledge = pd.pivot_table(data=pledge, index=['acc_id', "day"], aggfunc=sum)

	# 혈맹 feature rename
	rename_dict = {"etc_cnt" : "p_etc_cnt", "random_attacker_cnt" : "p_random_attacker_cnt",
              'same_pledge_cnt' : 'p_same_pledge_cnt', "temp_cnt" : "p_temp_cnt", 
               "random_defender_cnt" : "p_random_defender_cnt"}
	pledge.rename(columns=rename_dict, inplace=True)
	return pledge

def preprocessing_trade(trade, acc_id_combat):
	print('preprocessing trade table')
	trade.rename(columns={'source_acc_id':'acc_id'}, inplace=True) # rename source_acc_id

	# 같은 계정이면 같은 네트워크 그룹을 가질 확률이 높음 : 아이디 하나로 묶기
	trade.drop(["source_char_id", "target_char_id"], axis=1, inplace=True)
	# item_price feature NaN 중간값으로 치환
	trade["item_price"].fillna(trade["item_price"].median(), inplace=True)
	
	# Categorize transaction time
	# categroial로
	bins = [0, 60000, 120000, 180000, 239999]
	bin_label = [0, 1, 2, 3]   
	trade["time_bin"] = pd.cut(pd.to_numeric(trade["time"].str.replace(":", "")), bins=bins, labels=bin_label)
	trade.drop("time", axis=1, inplace=True) # 기존 time 은 제거
	trade['time_bin'] = trade['time_bin'].astype('category')
	trade['type'] = trade['type'].astype('category')

	# Total price per each trade
	# 현재는 median으로 대체된 값이 곱해져 있음을 기억해야 함
	trade["total_item_price"] = trade["item_amount"] * trade["item_price"]

	# source_id, target_id 에 대해 각각 dataframe 만들기
	source_trade = trade.drop("target_acc_id", axis=1)
	target_trade = trade.drop("acc_id", axis=1)
	target_trade.rename(columns={"target_acc_id" : "acc_id"}, inplace=True)

	# Total number of trade occurence per account the whole period
	source_trade_count = source_trade["acc_id"].value_counts()
	target_trade_count = target_trade["acc_id"].value_counts()

	# Make a dataframe to merge based on "acc_id"	
	source_trade_count = pd.DataFrame({"acc_id" : source_trade_count.index,
                                   "count" : source_trade_count})    
	target_trade_count = pd.DataFrame({"acc_id" : target_trade_count.index,
	                                   "count" : target_trade_count})

	trade_count = pd.merge(source_trade_count, target_trade_count, on="acc_id", how='outer')
	trade_count["count"] = trade_count["count_x"] + trade_count["count_y"]
	trade_count = trade_count.fillna(0)
	trade_count.drop(["count_x", "count_y"], axis=1, inplace=True)
	trade_count = trade_count.astype('int')
	# 현재 이건 오직 source acc_id data만 있는 것!
	source_trade = trade.drop(["target_acc_id", "item_type", "item_amount", "item_price"], axis=1)
	target_trade = trade.drop(["acc_id", "item_type", "item_amount", "item_price"], axis=1)
	target_trade.rename(columns={"target_acc_id" : "acc_id"}, inplace=True)
	# one-hot-encoding
	target_trade = pd.get_dummies(target_trade)
	source_trade = pd.get_dummies(source_trade)

	acc_id = acc_id_combat
	source_trade = pd.merge(source_trade,acc_id,on='acc_id',how='inner')
	target_trade = pd.merge(target_trade,acc_id,on='acc_id',how='inner')

	# (확인) time_bin 더해집니다.
	source_pivot = pd.pivot_table(data=source_trade, index=["acc_id", "day"], values=["type_0","type_1","time_bin_0","time_bin_1","time_bin_2","time_bin_3", "total_item_price"], aggfunc=sum)
	target_pivot = pd.pivot_table(data=target_trade, index=["acc_id", "day"], values=["type_0","type_1","time_bin_0","time_bin_1","time_bin_2","time_bin_3", "total_item_price"], aggfunc=sum)

	merged_trade = pd.merge(source_pivot, target_pivot, on=['acc_id','day'], how='outer').fillna(0)

	merged_trade['total_item_price'] = merged_trade['total_item_price_x'] + merged_trade['total_item_price_y']
	merged_trade['time_bin_0'] = merged_trade['time_bin_0_x'] + merged_trade['time_bin_0_y']
	merged_trade['time_bin_1'] = merged_trade['time_bin_1_x'] + merged_trade['time_bin_1_y']
	merged_trade['time_bin_2'] = merged_trade['time_bin_2_x'] + merged_trade['time_bin_2_y']
	merged_trade['time_bin_3'] = merged_trade['time_bin_3_x'] + merged_trade['time_bin_3_y']
	merged_trade['type_0'] = merged_trade['type_0_x'] + merged_trade['type_0_y']
	merged_trade['type_1'] = merged_trade['type_1_x'] + merged_trade['type_1_y']

	droplist = ['time_bin_0_x','time_bin_1_x','time_bin_2_x','time_bin_3_x','time_bin_0_y','time_bin_1_y','time_bin_2_y','time_bin_3_y',\
           'type_0_x','type_1_x','type_0_y','type_1_y','total_item_price_x','total_item_price_y']
	merged_trade = merged_trade.drop(columns=droplist)
	trade = merged_trade
	return trade

def preprocessing_payment(payment):
	# 아직 변경사항 없음
	print('preprocessing payment table')
	return payment