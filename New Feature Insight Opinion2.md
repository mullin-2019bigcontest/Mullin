## New Feature Insight

---

### Target Feature : 잔존기간, 결제액

---

### Feature Lists

- Server_activity => 잔존기간 or 결제액 높은 server 도출

  

- train_activity_total_sum

```python
train_activity_total_sum = sum(np.normalize(game_money_change & private shop & fishing & enchantcount & playtime))
```



- random_attacker_cnt & NPC kill( npc 죽인횟수 )

```python
attacker_npc = ("random_attacker_cnt" & "npc_kill")
```



- random_attacker_cnt & 특정 highest server 

```python
random_attacker_cnt & server(highest)
```



- Class & another Factors

  level 50 이전 기사 클래스 유저들의 접속률 & 거래금액 확인

  평균적인 기사/ 전사/ 요정/ 마법사 클래스의 평균금액이 전체적으로 높은지 확인

  ```python
  knight50 = (level < = 50) & (class = 1)
  np.mean(class = 1, class = 2, class = 3, class = 7) # 해당클래스들의 평균금액 높나?
  ```

  

- 캐릭터 사망 횟수 low / 플레이시간  (death / playtime)

```python
DeathPerTime = ("death" / "playtime") 
시간당 사망횟수가 적으면 strong user 확률 높아 
```



- boss_monster 보스 몬스터 타격 여부 (0=미타격 ,1= 타격) =>  총 타격 횟수 

```python
total_monter = value_count(boss_monster)
```



- game_money_change 일일 아데나 변동량 & enchant_count 7레벨 이상 아이템 인첸트 시도 횟수 high

```python
game_money_change 구간화,
enchant_count 구간화
=> 둘모두 높은 그룹의 결제액과 잔존일 예측
```

game_money_change 일일 아데나 변동량 AND enchant_count 7레벨 이상 아이템 인첸트 시도 횟수 high [구간화] (BY cut & bin) 상위 그룹에 대해 적용 진행 



- playtime 이 많은 user ID (activity) && (구매유저 / 판매) user id == > join ==>결제금액 예측

```python
playtime 구간화, 상위그룹에 해당 하는 userID 와
 join (target_acc_id OR source_acc_id)=> 예측
```



- pledge trade  time

  ```python
  group by 'pledge id', join.["tradetime"]
  ```

  거래발생시간이 비슷한 유저들 매칭(세분화).. 비슷한 시간대 존재하는지  ( 0 / 1 표현) => 1일시 그들의 잔존율, 결제액 확인 필요 

  tradetime 기존의 0,1,2,3 에서 더욱 세분화할 필요성...

  

- day_combat (전투날짜) && day_trade (거래 발생일) 

  전투를 하고 거래를 한날은 결제금액이 높은가 

  

- pledge_id 혈맹 아이디 && combat_char_cnt 

  전투 참여 혈맹 캐릭터 수 많으면 강직한 혈맹.

  ```python
  pledge_char_num = groupby["pledge_id"].combat_char_cnt
  ```

  

- same_char

평균 전투 참여 혈맹 캐릭터 수 많으면 강직한 혈맹....

=> 동일 혈맹, 평균 캐릭터당 혈맹 의 전투횟수 도출 

```python
groupby["same_pledge_cnt"].(combat_char_cnt / pledge_combat_cnt)
```

---



------

### Reference

- **Server_activity** 

**특화 서버 존재:** 리니지 초기의 치열했던 전장을 재현하기 위해 레벨 제한 및 PvP, 카오틱성향 유저 에 대한 페널티를 줄인 서버 (카오틱 = 공격적 막피 성향 유저 )

카오틱유저와같은 hard 유저들의 주무대 서버를 파악하고 평균 잔존기간 및 결제액확인



- **train_activity_total_sum** 

  **=>**  (0 ~ 1) 정규화된 / (**game_money_change & private shop & fishing & enchantcount & playtime**) 항목들의 수치를 **total_sum** 시킨다

 **'활동 이력'**을 데이터로 꼽는다면, 개발자는 유저가 게임에서 이탈할 시기가 가까워 질수록 게임 내 활동 이력이 줄어드는 것을 확인할 수 있다. 



- **random_attacker_cnt (막피를 행한 전투 횟수) & NPC kill( npc 죽인횟수 )**
- **random_attacker_cnt (막피를 행한 전투 횟수) & 특정 highest server **

막피를 거는 행위는 전투의 승리에 대한 자신감을 대표하는 행위임과 동시에,  총체적인 active한 활동지표중 하나로, 활동적이고 진취적인 hard user의 대표적행위이므로 해당분야 절대치뿐만아니라

**다른 hard user factor와 교집합되는 feature 생성시, 잠재적으로, 가장 주요한 요소가 될수 있음**

막피 횟수 => 막피 횟수가 높은 특정 서버를 알아낼수 있음 ... hard 막피유저 확률 , 거래량 및 접속률 높을 것 

**random_defender_cnt 적을수록 ... 강자 => hard user**



- **Class & another Factors**

**기사**

**=>** 레벨 50이전의 유저들중 거래횟수가 높은 유저들은 그기간내에 동일 레벨의 다른 캐릭터 유저들보다접속률 및 거래금액이 높을것

**level 50 이전 기사 클래스 유저들의 접속률 & 거래금액 확인** 

**=>** 가장 선형적인 캐릭터로 돈과 시간에 대한 신뢰도가 높은 클래스 

**전사**

**=>** 두배로 무기값이 든다는 점과 기사와 같이 겉과 속이 많이 다르다는점

**요정**

**=>** 개인의 장비나 스킬에 대한 금액 투자보다 시장을 조정하는 캐릭터라볼수 있음

**마법사**

**=>** 밸런스가 잘잡힌 클래스로, 이탈률이 높지않을것으로 예측

**평균적인 기사/ 전사/ 요정/ 마법사 클래스의 평균금액이 전체적으로 높은지 확인** 



- **boss_monster 보스 몬스터 타격 여부** (0=미타격 ,1= 타격) =>  **총 타격 횟수** 



-  **캐릭터 사망 횟수 low / 플레이시간**  (death / playtime)

  **=> 사망횟수 high 이면, 탈주율 높아 ... 반대 feature 생성,** 

시간당 사망횟수 적은 사람이 탈주율이 낮다고 볼수 있음 



- **game_money_change 일일 아데나 변동량** & **enchant_count 7레벨 이상 아이템 인첸트 시도 횟수 high**  ( 렙6 까지는 잘돼 )

우발적으로 강화이후 재 강화를 위해 돈을 쓰는 비율 높을수 있기에 => hard user 가능성 높다



- **playtime 이 많은 user ID (activity) && (구매유저 / 판매) user id == > join ==>** 

  결제금액 예측

  

- **pledge trade  time**

  **혈맹 아이디(pledge id) 가 동일한 유저들 그룹화... 거래발생시간(time_trade) 비슷한 유저들 매칭.... 지속적인 시간대 (time bin) 존재하는지** ==>  1/0 표현 

  

- **game_money_change 아데나 변동량 큰 day 날 & train_day 거래발생일 => 해당일 결제금액 예측** ( 특정일 금액 지출 유무확인 )



- **day_combat (전투날짜) join day_trade (거래 발생일 )** 

  전투를 하고 거래를 한날은 결제금액이 높은가 

  

- **pledge_id 혈맹 아이디 && combat_char_cnt** 

  전투 참여 혈맹 캐릭터 수 많으면 강직한 혈맹.

  

- **same_pledge_cnt 동일혈맹 전투횟수 / combat_char_cnt**

  전투 참여 혈맹 캐릭터 수

**=>** **평균 한 캐릭터당 동일혈맹 전투참여수** 



- **혈맹 아이디 && 전투 참여 혈맹 캐릭터 수 많으면 강직한 혈맹....**

combat_play_time 전투 캐릭터 플레이 시간의 합 / combat_char_cnt 전투 참여 혈맹 캐릭터 수

=> **평균 혈맹 캐릭터당 혈맹 의 플레이시간 도출** 





- **day_combat (전투날짜) join day_trade (거래 발생일 )** 

  전투를 하고 거래를 한날은 결제금액이 높은가 

  
