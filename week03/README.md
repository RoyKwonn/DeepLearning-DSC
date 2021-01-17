# pyCharm 설정 (우분투에서)

## 1. 다운로드 및 설치 


### 1.1. 다운로드
pyCharm 홈페이지에서 community버전을 다운로드를 받는다.
https://www.jetbrains.com/pycharm/


### 1.2. 설치하기

다운받은 파일 압축해제 후 아래 폴더로 들어갑니다.

```
$ cd ~/다운로드/pycharm-community-2020.3.2/bin
$ sh pycharm.sh
```

or

```
$ ~/download/pycharm-community-2020.3.2/bin
$ sh pycharm.sh
```

## 2 Create Desktop Entry
어떤 곳에서도 pycharm.sh에 접근할 수 있도록 아래 절차를 따라해주세요.

상단 탭에서
Tools -> Create Desktop Entry... 를 클릭합니다.
체크박스가 나오는데 체크를 한다음 ok를 눌러줍니다


## 3. pychar에 Anaconda 설정 적용하기

1. 상단탭에서 File -> Settings
2. 왼쪽 메뉴에서 Project : XXXXX -> Python Interpreter -> py37를 적용 -> OK



