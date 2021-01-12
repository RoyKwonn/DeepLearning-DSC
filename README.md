# DeapLearning-DSC

## INSTALL UBUNTU Desktop 18.04.5 (Virtual machine OR Multi boot)
 
### 1. DOWNLOAD UBUNTU 

You shuld download with torrent because of downloading rapidlly

https://ubuntu.com/download/alternative-downloads

Virtual box에 설치한 경우
```
$ sudo apt-get install build-essential linux-headers-$(uname -r)
```

### 2. 프로그램 충돌방지 

업데이트를 하자
```
$ sudo apt-get update && upgrade
$ sudo apt dist-upgrade
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

### 4. Install anaconda3-2020.02

Keep in mind that the version 2020.02 that is mentioned in the above command is only compatible with Python 3.7. So, if you are operating the 2.7 version of Python, then use the link mentioned above to find the compatible version of Anaconda with your Python 2.7 version.

```
$ pyenv install anaconda3-2020.02
$ git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
$ echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
$ source ~/.bash_profile
```

### 5. set virtualenv

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

## 6. 가상화 종료
```
$ pyenv deactivate
```




