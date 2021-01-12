# DeapLearning-DSC

## INSTALL UBUNTU Desktop 18.04.5 (Virtual machine OR Multi boot)
 
### 1. DOWNLOAD UBUNTU 

You shuld download with torrent because of downloading rapidlly

https://ubuntu.com/download/alternative-downloads

### 2. 

업데이트를 하자
```
$ sudo apt-get update && upgrade
```

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

잘 설치되어있는지 확인해보자.
```
$ pyenv versions
```

