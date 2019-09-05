# '깃허브로 협업하기' 튜토리얼

<br>

## TL; DR

명령어 대괄호 부분만 수정후, git bash 창에 그대로 복붙하면 됩니다.

<u>**Step1. 작업 공간 만들기**</u>

1. (my 원격 저장소 생성) [멀린-빅콘테 깃허브 중앙 원격 저장소](https://github.com/mullin-2019bigcontest/Mullin) 에서 `Fork` 한다.
2. (my 로컬 저장소 생성) 생성된 my 원격 저장소로부터 내 컴퓨터로 복제 한다. `git clone https://github.com/[__username__]/Mullin`
3. my 로컬 저장소와 중앙 원격 저장소를 연결 짓는다. `git remote add upstream https://github.com/mullin-2019bigcontest/Mullin`
4. my 로컬 저장소에서 작업할 branch 를 만들고 해당 branch 로 이동한다. `git checkout -b [branch name]`
5. 생성한 branch 에서 작업한다.

<u>**Step2. 중앙 원격 저장소로 작업 내용 적용시키기**</u>

1. my 로컬 저장소의 브랜치에서 my 원격 저장소로 push 한다. `git push origin [branch name]`
2. my 원격 저장소 깃허브 페이지로 들어가서 `pull request` 한다.
3. 중앙 원격 저장소에서 `merge` 한다.
4. (merge가 됐다면) 중앙 원격 저장소와 my 로컬 저장소를 동기화 한다. `git checkout master; git pull upstream master`
5. Step1 의 4.~5. 반복하며 작업 진행한다.

<br>

<br>

## Forking Workflow 방식으로 협업하기

깃허브로 협업하는 방법에는

- Feature Branch Workflow
- Forking Workflow
- Gitflow Workflow

이렇게 3가지가 있는데, 큰 규모의 프로젝트 및 오픈 소스 프로젝트에서 많이 사용한다는 **Forking Workflow** 방식을 선택해봤어요.

Forking Workflow 방식은 팀원 모두가 로컬 저장소, fork된 원격 저장소 하나씩을 갖고 협업을 진행하는 방식입니다.

[중앙 원격 저장소](https://github.com/mullin-2019bigcontest/Mullin)로 바로 푸시하는 것이 아니라 자신이 소유한 fork된 원격 저장소에서 중앙 원격 저장소로 **Pull Request** 하며 업데이트합니다.

<br>

<br>

## Step1. 작업 공간 만들기

<br>

[멀린-빅콘테 깃허브 중앙 원격 저장소](https://github.com/mullin-2019bigcontest/Mullin)에 들어가 **fork** 버튼 클릭!

![](/etc/fork.png)

<br>

그러면 자동으로 Mullin 이름의 repository 가 계정에 만들어집니다.

![](/etc/forked.png)

<br>

해당 저장소를 clone 합니다.

`Clone or download` 버튼 클릭해서 remote repository URL 복사해둡니다.

![](/etc/clone_copy.png)

<br>

git bash 창에서

```powershell
$ git clone [remote repository URL]
```

실행.

이제 내 컴퓨터에 로컬 저장소가 생성됐습니다.

<br>

이제 중앙 원격 저장소와 내 로컬 저장소를 연결해야 합니다.

```powershell
$ git remote add upstream https://github.com/mullin-2019bigcontest/Mullin
```

**<u>설명</u>**

현재 내 로컬 저장소에서 `https://github.com/mullin-2019bigcontest/Mullin` 주소와 연결 시키는데, **upstream** 이라는 이름으로 연결시킨다.

<br>

이제

```powershell
$ git remote -v
```

명령어를 실행해보면 **origin** 이라는 이름으로 내 원격 저장소가 연결돼 있고, **upstream** 이라는 이름으로 중앙 원격 저장소가 연결돼 있음을 알 수 있습니다.

(**origin** 이라는 이름은 clone 했을 시 깃에서 자동으로 지어주는 원격 저장소 이름이에요.)

> 이제 `git push upstream master` 라는 명령어로 중앙 원격 저장소에 푸시할 수 있지만, 이는 Forking Workflow 방식에 위배됩니다. 금지!

<br>

<참고 그림>

![](/etc/upstream_origin.png)

<br>

이제 브랜치를 하나 생성 후 그 곳에서 작업하려 합니다.

여기서 master 브랜치가 아니라 새 브랜치를 만드는 이유는, 명시적으로 로컬 저장소의 master 브랜치는 중앙 원격 저장소의 master 브랜치와 동일함을 표현하기 위해서에요.

git bash에서 아래 명령어를 실행해  브랜치를 만들고 해당 브랜치로 이동합니다.

```powershell
$ git checkout -b [branch name] # 이 명령어는 아래 두 명령어를 합친것

$ git branch [branch name] # 새로운 브랜치 생성
$ git checkout [branch name] # 해당 브랜치로 작업 위치 이동
```

<br>

이제

```powershell
$ git branch # 현재 위치한 브랜치 및 모든 브랜치 표시
```

명령어를 실행해보거나, `git status` 명령어를 실행해보세요.

현재 위치한 branch 를 알 수 있습니다. <u>**꼭 확인 후 작업!!**</u>

<br>

이제 ***브랜치가 뭔지 이해***해보기 위해 아래 명령어들을 실행해 봅시다.

```powershell
$ touch fileonbranch.txt # 빈 .txt 파일 생성
$ git add fileonbranch.txt # .txt 파일 스테이징
$ git commit -m "연습" # 커밋
```

이제 txt 파일이 하나 생성돼 있을텐데, 여기서 원래의 master 브랜치로 이동해볼게요.

```powershell
$ git checkout master # master branch 로 이동
```

master 브랜치로 이동한 순간 폴더 내에 fileonbranch.txt 파일이 사라져 있음을 알 수 있습니다.

master 브랜치 커밋 내용엔 없는 파일이기 때문입니다. 이처럼 새 브랜치를 만들어서 작업하면 중앙 원격 저장소와 동기화 돼 있는 로컬의 master 브랜치는 보존한 채로 작업할 수 있어요.

<br>

<br>

## Step2. 중앙 원격 저장소로 작업 내용 적용시키기

<br>

작업이 진행되고 해당 작업 내용에대해 팀원들과 협의가 완료됐다면 이제 완성본만 저장이되는 중앙 원격 저장소로 코드를 옮길 차례입니다.

그런데 Forking Workflow 방식에서는 로컬에서 바로 푸시하는것이 아니라 fork 된 원격 저장소에서 중앙 원격 저장소로 **<u>Pull Request</u>** 하는 방식을 따릅니다.

<br>

일단 작업중인 브랜치 이름을 `joyoon` 이라고 할때,

```powershell
$ git checkout joyoon # joyoon 브랜치로 이동
$ git add .
$ git commit -m "커밋"
$ git push origin joyoon # joyoon 브랜치의 내용을 원격 저장소 origin 으로 푸시
```

이 명령어들을 통해 나의 원격 저장소로 `joyoon` 이라는 브랜치 이름으로 푸시할 수 있습니다.

깃허브 페이지에서는

![](/etc/branch.png)

이렇게 들어가서 확인할 수 있습니다.

<br>

이제 브랜치 내용을 Pull Request 해서 중앙 원격 저장소에 적용해달라고 요청을 합니다.

![](/etc/pull_request.png)

<br>

그럼 [깃허브 중앙 원격 저장소 페이지의 Pull Requests 페이지](https://github.com/mullin-2019bigcontest/Mullin/pulls)에 해당 요청이 업데이트 된걸 확인할 수 있습니다. 여기서 요청을 확인하고 merge 하면 중앙 원격 저장소로의 적용이 완료가 됩니다.

<br>

<br>

<br>

---

<br>

## Reference

[https://andamiro25.tistory.com/193](https://andamiro25.tistory.com/193)

[https://milooy.wordpress.com/2017/06/21/working-together-with-github-tutorial/](https://milooy.wordpress.com/2017/06/21/working-together-with-github-tutorial/)