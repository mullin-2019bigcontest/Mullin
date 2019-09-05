import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 수정완료
def preprocessing_activity(activity):
    print('preprocessing activity table')
    activity.drop('char_id', axis=1, inplace=True)
    total_play = activity.groupby('acc_id')['playtime'].sum()
    total_play = pd.DataFrame({'total_play_time': total_play}) # merge 위해 생성
    activity = pd.pivot_table(data=activity.drop('server', axis=1), index=['acc_id', 'day'], aggfunc=sum) # 하루동안 모든 value의 합
    activity["total_exp"] = activity["solo_exp"] + activity["party_exp"] + activity["quest_exp"] # 총 exp 획득량
    activity["fishing_prop"] = activity["fishing"] / activity["playtime"] # 하루에 낚시에 투자하는 비율
    activity['qexp_per_playtime'] = activity['quest_exp'] / activity['playtime']
    activity['sexp_per_playtime'] = activity['solo_exp'] / activity['playtime']
    activity['pexp_per_playtime'] = activity['party_exp'] / activity['playtime']
    activity['activity_logged_in'] = 1
    return activity

# 수정완료
def preprocessing_combat(combat):
    print('preprocessing combat table')
    combat.drop(['char_id', 'server'], axis=1, inplace=True)
    combat['class'] = combat['class'].astype('category') # 'class' feature 카테고리화
    prop_class = combat['class'].value_counts() / combat['class'].value_counts().sum() # 클래스별 전체 비율

	# 전체 누적합 60% 미만을 차지하는 주류 클래스 조사
    class60 = prop_class[~(prop_class.cumsum()>0.6)].index # 기사, 요정, 마법사 (70%로 하면 전사까지.)
    combat['isMajorClass'] = combat['class'].apply(lambda x: 1 if x in class60 else 0) 

    combat = combat.rename(columns={"class":"c_cnt_class"})
    
    # max_level = combat.groupby('acc_id')['level'].max() # 계정의 최대 레벨
	# mean_level = combat.groupby('acc_id')['level'].mean() # 계정의 평균 레벨
    combat['maxlevel'] = combat.groupby("acc_id")["level"].transform(lambda x:x.max())
    combat['meanlevel'] = combat.groupby("acc_id")["level"].transform(lambda x:x.mean())
    combat['minlevel'] = combat.groupby('acc_id')['level'].transform(lambda x:x.min())
    combat['totlevel'] = combat.groupby('acc_id')['level'].transform(lambda x:x.sum())
    combat.drop('level', axis=1, inplace=True) # 기존 level feature drop
    combat['combat_logged_in'] = 1

    combat = pd.get_dummies(combat) # class feature 에 대해 get_dummies 적용됨

	# 한 계정에 대해 주류 클래스만 플레이 했을경우 1.0 아니면 비주류 플레이한 만큼 지표(평균) 떨어짐
    combat = pd.concat([pd.pivot_table(data=combat.drop('isMajorClass', axis=1), index=["acc_id", "day"], aggfunc=sum), pd.pivot_table(data=combat, index=['acc_id', 'day'], values='isMajorClass', aggfunc='mean')], axis=1)
    return combat

# 혈맹은 join시 보간을 따로 해줘야 함 (혈맹이 바뀌진 않으니까)
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
    pledge_num_people = pledge["pledge_id"].value_counts().reset_index()
    pledge_num_people = pd.DataFrame(pledge_num_people)
    pledge_num_people = pledge_num_people.rename(columns={"index":"pledge_id","pledge_id":"pledge_num_people"})

    pledge = pd.merge(pledge,pledge_num_people,on='pledge_id',how='left')

    pledge.drop(["pledge_id"], axis=1, inplace=True)
    pledge.drop(['char_id'], axis=1, inplace=True)

    pledge = pd.concat([pd.pivot_table(data=pledge.drop(columns='pledge_num_people'), index=['acc_id', "day"], aggfunc=sum),\
          pd.pivot_table(data=pledge[['acc_id', "day",'pledge_num_people']], index=['acc_id', "day"], aggfunc='mean')],\
         axis=1)
    pledge['pledge_logged_in'] = 1
	# 혈맹 feature rename
    rename_dict = {"etc_cnt" : "p_etc_cnt", "random_attacker_cnt" : "p_random_attacker_cnt",
              'same_pledge_cnt' : 'p_same_pledge_cnt', "temp_cnt" : "p_temp_cnt", 
               "random_defender_cnt" : "p_random_defender_cnt"}
    pledge.rename(columns=rename_dict, inplace=True)
    return pledge

def preprocessing_trade(trade, acc_id_combat):
    print('preprocessing trade table')
    trade.rename(columns={'source_acc_id':'acc_id'}, inplace=True) # rename source_acc_id

    # item_type : 숫자 라벨링
    itemtype_list = trade['item_type'].unique()
    le = LabelEncoder()
    le.fit(itemtype_list)
    trade['item_type'] =  le.transform(trade['item_type'])

	# 같은 계정이면 같은 네트워크 그룹을 가질 확률이 높음 : 아이디 하나로 묶기
    trade.drop(["source_char_id", "target_char_id"], axis=1, inplace=True)
	# # item_price feature NaN 중간값으로 치환
	# trade["item_price"].fillna(trade["item_price"].median(), inplace=True)
	
	# Categorize transaction time
	# categroial로
    bins = [0,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000\
        ,160000,170000,180000,190000,200000,210000,220000,230000,239999]
    bin_label = range(0,24,1)
    trade["time_bin"] = pd.cut(pd.to_numeric(trade["time"].str.replace(":", "")), bins=bins, labels=bin_label)
    trade.drop("time", axis=1, inplace=True) # 기존 time 은 제거
    trade.drop("server", axis=1, inplace=True) # 서버 제거 

    trade['time_bin'] = trade['time_bin'].fillna(0)
    trade['time_bin'] = trade['time_bin'].astype('category')
    trade['type'] = trade['type'].astype('category')
    trade['item_type'] = trade['item_type'].astype('category')

	# Total price per each trade
	# 현재는 median으로 대체된 값이 곱해져 있음을 기억해야 함
	# trade["total_item_price"] = trade["item_amount"] * trade["item_price"]

    # 정규화된 값을 다시 adena로 환산
    trade_item_amount_rev = trade['item_amount']/trade['item_amount'].min()
    trade_item_price_rev = trade['item_price']/trade['item_price'].min()

    # 그 값을 trade에 넣어줌
    trade['item_amount'] = trade_item_amount_rev
    trade['item_price'] = trade_item_price_rev
    trade['item_price'] = np.where(trade['item_type']==1, trade['item_amount'], trade['item_price'])
    trade['item_amount'] = np.where(trade['item_type']==1, 1, trade['item_amount'])

    # # na값들 전부 drop (price가 NA인 값들은 전부 drop) : 
    # trade = trade.dropna()

    # 아이템 타입의 평균으로 보간

    trade['item_price'] = trade.groupby('item_type')['item_price'].transform(lambda x: x.fillna(x.mean()))
    trade["total_item_price"] = trade["item_amount"] * trade["item_price"]

	# source_id, target_id 에 대해 각각 dataframe 만들기
    source_trade = trade.drop("target_acc_id", axis=1)
    target_trade = trade.drop("acc_id", axis=1)
    target_trade.rename(columns={"target_acc_id" : "acc_id"}, inplace=True)

    # 각 아이템을 몇번 판매했는지 count / 각 아이템을 얼마에 판매했는지 amount
    itemtype_list = ['accessory', 'adena', 'armor', 'enchant_scroll', 'etc', 'spell','weapon']
    for type in itemtype_list:
        source_trade['sell_amount_'+type] = 0
        source_trade['sell_count_'+type] = 0

    source_list = ['sell_count_accessory', 'sell_count_adena', 'sell_count_armor', 'sell_count_enchant_scroll',\
       'sell_count_etc', 'sell_count_spell', 'sell_count_weapon']
    source_dict = {x:i for x,i in zip(source_list,range(0,7))}
    for source in source_list:
        source_trade[source] = np.where(source_trade['item_type']==source_dict[source], source_trade[source]+source_trade['item_amount'],source_trade[source])

    source_list = ['sell_amount_accessory', 'sell_amount_adena', 'sell_amount_armor', 'sell_amount_enchant_scroll',\
       'sell_amount_etc', 'sell_amount_spell', 'sell_amount_weapon']
    source_dict = {x:i for x,i in zip(source_list,range(0,7))}
    for source in source_list:
        source_trade[source] = np.where(source_trade['item_type']==source_dict[source], source_trade[source]+(source_trade['total_item_price']),source_trade[source])

    source_trade = source_trade.rename(columns={"server":"sell_server", "type":"sell_type","total_item_price":"total_sell_item_price","item_amount":"sell_item_amount"\
                                           ,"item_type":"sell_item_type", "item_price":"sell_item_price","time_bin":"sell_time_bin"})

    # 각 아이템을 몇번 구매했는지 count / 각 아이템을 얼마에 구매했는지 amount
    itemtype_list = ['accessory', 'adena', 'armor', 'enchant_scroll', 'etc', 'spell','weapon']
    for type in itemtype_list:
        target_trade['get_amount_'+type] = 0
        target_trade['get_count_'+type] = 0

    target_list = ['get_count_accessory', 'get_count_adena', 'get_count_armor', 'get_count_enchant_scroll',\
       'get_count_etc', 'get_count_spell', 'get_count_weapon']
    target_dict = {x:i for x,i in zip(target_list,range(0,7))}

    for target in target_list:
        target_trade[target] = np.where(target_trade['item_type']==target_dict[target], target_trade[target]+target_trade['item_amount'],target_trade[target])


    target_list = ['get_amount_accessory', 'get_amount_adena', 'get_amount_armor', 'get_amount_enchant_scroll',\
       'get_amount_etc', 'get_amount_spell', 'get_amount_weapon']
    target_dict = {x:i for x,i in zip(target_list,range(0,7))}

    for target in target_list:
        target_trade[target] = np.where(target_trade['item_type']==target_dict[target], target_trade[target]+target_trade['total_item_price'],target_trade[target])

    target_trade = target_trade.rename(columns={"type":"get_type","server":"get_server","total_item_price":"total_get_item_price",'item_amount':'get_item_amount'\
                                           ,"item_type":"get_item_type",'item_price':"get_item_price","time_bin":"get_time_bin"})
    
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
	# trade_count.drop(["count_x", "count_y"], axis=1, inplace=True)
    trade_count = trade_count.rename(columns={'count':'total_trade_count', 'count_x':'count_sell', 'count_y':'count_get'})
    trade_count = trade_count.astype('int')
	# 현재 이건 오직 source acc_id data만 있는 것!
	# source_trade = trade.drop(["target_acc_id", "item_type", "item_amount", "item_price"], axis=1)
	# target_trade = trade.drop(["acc_id", "item_type", "item_amount", "item_price"], axis=1)
	# target_trade.rename(columns={"target_acc_id" : "acc_id"}, inplace=True)
	# one-hot-encoding
    target_trade = pd.get_dummies(target_trade)
    source_trade = pd.get_dummies(source_trade)


    acc_id = acc_id_combat
    source_trade = pd.merge(source_trade,acc_id,on='acc_id',how='inner')
    target_trade = pd.merge(target_trade,acc_id,on='acc_id',how='inner')

	# (확인) time_bin 더해집니다.
    source_pivot = pd.pivot_table(data=source_trade, index=["acc_id", "day"], aggfunc=sum)
    target_pivot = pd.pivot_table(data=target_trade, index=["acc_id", "day"], aggfunc=sum)

    merged_trade = pd.merge(source_pivot, target_pivot, on=['acc_id','day'], how='outer').fillna(0)
    merged_trade = merged_trade.fillna(0)

    selllist = ['sell_amount_accessory',
    'sell_amount_adena',
    'sell_amount_armor',
    'sell_amount_enchant_scroll',
    'sell_amount_etc',
    'sell_amount_spell',
    'sell_amount_weapon',
    'sell_item_amount']
    selllist_ = [ 'sell_count_accessory',
    'sell_count_adena',
    'sell_count_armor',
    'sell_count_enchant_scroll',
    'sell_count_etc',
    'sell_count_spell',
    'sell_count_weapon']
    getlist = ['get_amount_accessory',
    'get_amount_adena',
    'get_amount_armor',
    'get_amount_enchant_scroll',
    'get_amount_etc',
    'get_amount_spell',
    'get_amount_weapon',
    'get_item_amount']
    getlist_ = [ 'get_count_accessory',
    'get_count_adena',
    'get_count_armor',
    'get_count_enchant_scroll',
    'get_count_etc',
    'get_count_spell',
    'get_count_weapon']
    newlist = ['tot_amount_accessory',
    'tot_amount_adena',
    'tot_amount_armor',
    'tot_amount_enchant_scroll',
    'tot_amount_etc',
    'tot_amount_spell',
    'tot_amount_weapon',
    'tot_item_amount']
    newlist_ = [ 'tot_count_accessory',
    'tot_count_adena',
    'tot_count_armor',
    'tot_count_enchant_scroll',
    'tot_count_etc',
    'tot_count_spell',
    'tot_count_weapon']

    for sell, get, new in zip(selllist, getlist, newlist):
        merged_trade[new] = merged_trade[sell] - merged_trade[get]
    
    for sell, get, new in zip(selllist_, getlist_, newlist_):
        merged_trade[new] = merged_trade[sell] + merged_trade[get]
    
    for i in range(7):
        merged_trade['tot_item_type_'+str(i)] = merged_trade['sell_item_type_'+str(i)] + merged_trade['get_item_type_'+str(i)]

    for i in range(24):
        merged_trade['tot_time_bin_'+str(i)] = merged_trade['sell_time_bin_'+str(i)] + merged_trade['get_time_bin_'+str(i)]

    merged_trade['trade_logged_in'] = 1
    
    trade = merged_trade
    return trade

def preprocessing_payment(payment):
	# 내용 추가
    print('preprocessing payment table')
    payment['max_spent'] = payment.groupby('acc_id')['amount_spent'].transform(lambda x:x.max())
    payment['min_spent'] = payment.groupby('acc_id')['amount_spent'].transform(lambda x:x.min())
    payment['mean_spent'] = payment.groupby('acc_id')['amount_spent'].transform(lambda x:x.mean())
    payment['tot_spent'] = payment.groupby('acc_id')['amount_spent'].transform(lambda x:x.sum())
    payment['median_spent'] = payment.groupby('acc_id')['amount_spent'].transform(lambda x:x.median())
    payment = payment.set_index(['acc_id','day'])
    payment['payment_logged_in'] = 1
    return payment