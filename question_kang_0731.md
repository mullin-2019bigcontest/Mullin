# 2019/07/31

### 데이터 분석 피드백 (및 질문)

#### - 각 file에 대해 isin(label)을 한 이유가 궁금합니다.

#### - all_char의 의미가 궁금합니다.

#### 	기존 방식으로 모으면 '캐릭터의 수'가 아니라 '활동'(combat/pledge/activity)의 수	가 모아진 것이 아닐까요?

#### 	all_char.groupby('acc_id')['char_id'].nunique() 해줘야 하는게 아닐까요?

#### - 각 table에서 server가 다른데 같은 acc_id를 가진 값들이 있는데 분리가 잘 안되어 있는건 아닐까요..?