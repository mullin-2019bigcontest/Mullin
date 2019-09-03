## New Feature Insight

---



### Target Feature : 잔존기간, 결제액

---

## 수식 으로!! 

## 코딩 까지!! 맨들어!! 



### Feature Lists

- **Server_activity**

커뮤니티 사이에서 '데스나이트'와 같은 서버는 일명 "라인"사람들로 명명되는 hard user들의 주 활동무대라고 알려져있어 그 반대인 "중립"으로 불리는 light user들은 기피하는 server이다 

고로, hard user 들이 선호하는 server 파악 및 상관관계분석이 필요할 전망

---

- **quest_exp 퀘스트 획득 경험치 / playtime 플레이시간**



- **train_activity_total_sum**

게임 내 전체 활동 train_activity의 절대합 횟수에 따른 TargetFeature의 영향력 차이 상관관계분석필요

 **'활동 이력'**을 데이터로 꼽는다면, 개발자는 유저가 게임에서 이탈할 시기가 가까워 질수록 게임 내 활동 이력이 줄어드는 것을 확인할 수 있다. 

고로, 해당 테이블의  전체 합 또한 주요 상관관계를 지닐수 있다 

---



- **random_attacker_cnt (막피를 행한 전투 횟수) & high level**

막피를 거는 행위는 전투의 승리에 대한 자신감을 대표하는 행위임과 동시에,  총체적인 active한 활동지표중 하나로, 활동적이고 진취적인 hard user의 대표적행위이므로 해당분야 절대치뿐만아니라

**다른 hard user factor와 교집합되는 feature 생성시, 잠재적으로, 가장 주요한 요소가 될수 있음**

---



- **Non_temp_cnt ( 비단발성전투 )**

**단발성 전투** 는 "전투한 캐릭터 수가 일정 기준 이하인 캐릭터 간의 전투" 를 뜻하므로 비단발성 전투는

캐릭터수를 다수보유한 active user의 특성이 될수 있을 전망

---



- **Class**
- **Class & another Factors**

기본적으로 classical class 에 해당하는 기사 요정 마법사 클래스에 국한하여 설명할시, 

기사나 요정은 마법사 클래스에 비해 투자대비? 만족도 높은 성과가 배출됨으로 hard user들은 이러한 기사, 요정에 대한 충성도가 높을수 있다 

대표적으로, 클래스 선호도에 따른 접근, 기사나 요정은 마법사 클래스에 비해 투자대비? 만족도 높은 성과가 배출됨으로 hard user들은 이러한 기사, 요정에 대한 충성도가 높을수 있다.

기사의 경우 방어구를 더 선호 (방어 캐릭)

다크엘프와 요정이 무기를 조금 더 선호 (공격 캐릭)

- boss_monster 보스 몬스터 타격 여부 (0=미타격 ,1= 타격) =>  **총 타격 횟수** 

-  **캐릭터 사망 횟수 / 플레이시간**  (death / playtime)

  => 탈주율 높아 ... 반대 feature 생성, 

시간당 사망횟수 적은 사람이 탈주율이 낮다고 볼수 있음 



- **game_money_change 일일 아데나 변동량** & **enchant_count 7레벨 이상 아이템 인첸트 시도 횟수**



- playtime 이 많은 user ID (activity) && 구매유저 / 판매 user id == > join ==> 결제금액 예측

  

- 혈맹 아이디(pledge id) 가 동일한 유저들 그룹화... 거래발생시간(time_trade) 비슷한 유저들 매칭.... 지속적인 시간대 (time bin) 존재하는지 ==> 

  

- **random_defender_cnt 적을수록 ... 강자 => hard user**

- **random_attacker_cnt  많을수록... 강장 => hard user** 







---

---



#### References



- **Server** 

**PVP서버:** 말 그대로 PVP, 즉 대인전이 가능한 서버. 대부분의 서버가 PVP서버에 해당한다.

**Non-PVP:** 캐릭터 간의 PVP가 허용되지 않는 서버

**특화 서버:** 리니지 초기의 치열했던 전장을 재현하기 위해 레벨 제한 및 PvP,

 [카오틱 성향](https://lineage.plaync.com/powerbook/wiki/카오틱+성향)에 대한 페널티를 줄인 서버 (카오틱 = 공격적 막피 성향 유저 )



- **party_exp**

 파티원의 레벨이 레벨이 높을수록 획득할 수 있는 경험치의 양이 감소하는 것을 확인할 수 있었다



- Class & Item Type

기사의 경우 방어구를 더 선호하는 것으로 보입니다. 

나머지 직업에 대해서는 통계적으로 유의한 차이는 아니지만, 다크엘프와 요정이 무기를 조금 더 선호하는 것으로 보입니다. 

실제로 기사는 다른 클래스에 비해 근거리 데미지와 체력이 높은 방어 캐릭터로 유명하고 다크엘프와 요정은 공격형 캐릭터로 분류되죠.

---

---



ref ) https://danbi-ncsoft.github.io

ref ) [http://rigvedawiki.net/w/%EB%A6%AC%EB%8B%88%EC%A7%80%28%EA%B2%8C%EC%9E%84%29](http://rigvedawiki.net/w/리니지(게임))
