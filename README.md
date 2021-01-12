# DeapLearning-DSC

## 개발환경 
 
### 0. DOWNLOAD UBUNTU 

You shuld download with torrent because of downloading rapidlly

https://ubuntu.com/download/alternative-downloads


### 1. INSTALL UBUNTU Desktop 18.04.5 (Virtual machine OR Multi boot)

#### 1.1. HOST 설정
- 제어판 -> Windows 기능 켜기/끄기 -> Hyper-V 해제    // 없으면 안해도 됨
- (BIOS 모드) Advanced Mode -> Intel VT-x 또는 AMD-V Enable   // 맥은 안해도됨

#### 1.2. VM
- 머신 -> 새로만들기
- 메모리 크기 지정 (2048 MB 이상)

VM-하드디스크 생성
- 지금 새 가상 하드 디스크 만들기
- VDI (VirtualBox 디스크 이미지)
- 고정 크기
- 파일 위치 및 크기 (파일크기 30 GB 이상)

VM-설정 (언급된건 체크)
- 일반 항목 -> 클립보드, 드래그 앤 드롭 : 양방향 설정
- 시스템 항목 -> 마더보드 : 플로피 디스크 해제, ICH9 칩셋, EFI 사용하기
- 시스템 항목 -> 프로세서 : 프로세서 개수(실제 호스트 프로세서의 절반), PAN/NX 사용하기 
- 디스플레이 항목 -> 화면 : 비디오메모리(최대), 3차원 가속 사용하기
- 저장소 항목 -> CD아이콘(광학 드라이브) : 다운받은 UBUNTU_XXXX.ISO 선택

머신 실행
- 언어, 사용자 이름, 비번 설정
- 그외 항목은 디폴트 선택


터미널 실행
```
$ sudo apt-get update && upgrade
$ sudo apt dist-upgrade
$ sudo apt-get install build-essential linux-headers-$(uname -r)
```

VM-상단메뉴
- 장치 -> 게스트확장 CD 이미지 삽입
- 실행
- 설치완료 후 바탕화면의 CD 아이콘 오른쪽 마우스 클릭하여 꺼내기
- 재부팅



### 2. 프로그램 충돌방지 

Build할 때 공통적으로 발생하는 문제를 방지하기 위해 필요한 패키지들을 설치해준다.
```
$ sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev
```

### 3. pyenv

git이 설치되지 않았다면 설치해준다.
```
$ sudo apt install git
```

git clone을 이용하여 소스를 다운받고 몇 가지 설정을 해준다.
```
$ git clone https://github.com/pyenv/pyenv.git ~/.pyenv
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
$ echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
$ source ~/.bash_profile
```

### 4. Install anaconda3-2020.02

Keep in mind that the version 2020.02 that is mentioned in the above command is only compatible with Python 3.7. So, if you are operating the 2.7 version of Python, then use the link mentioned above to find the compatible version of Anaconda with your Python 2.7 version.

```
$ pyenv install anaconda3-2020.02
$ git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
$ echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
$ source ~/.bash_profile
```

### 5. set virtualenv

가상화 설정
```
$ pyenv virtualenv anaconda3-2020.02 deeplearning
$ pyenv versions
  anaconda3-2020.02
  anaconda3-2020.02/envs/deeplearning
  deeplearning
$ pyenv activate deeplearning
(deeplearning) $ python -v
...
```

가상화 종료
```
$ pyenv deactivate
```

### 6. Anaconda 설정

가상화 실행상태에서
```
(deeplearning) $ conda create -n py37 python=3.7
(deeplearning) $ conda activate py37
(py37)(deeplearning) $ pip install tensorflow==2.0.0
(py37)(deeplearning) $ pip install keras==2.3

```

혹시 바로 안된다면, 아래 명령어 실행후 다시 위에 작업을 해주어라
```
(py37)(deeplearning) $ pyenv deactivate deeplearning
```

아래와 같은 내용이 출력된다면 잘 설치한 것이다.
```
(py37)(deeplearning) $ python
Python 3.7.9 (default, Aug 31 2020, 12:42:55) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version__)
2.0.0
>>> import keras
Using TensorFlow backend.
>>> exit()
```

