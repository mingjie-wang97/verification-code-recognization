Giving Credit to the origional project
https://github.com/hellokuls/cnnyzm


All training Models are in
https://drive.google.com/drive/folders/18Smu7IAgkaSCwvJt29CREJ5gaWjgR95K?usp=sharing


Files Details --> 

generate.py generates the verification code

getimage.py reads all verification code image

process.py does some post-process to the verification code image

train.py does training for that (best situation is using 10,000 images)

test.py is the test tool

if we are using tensorflow 1.XX;
chagne 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior
to 
import tensorflow as tf


The latest model has a suceess reading rate over 90%


python virtual env set up process

pre-request for install venv on ubuntu
sudo apt-get install python3-venv

创建一个新的虚拟环境，方法是选择 Python 解释器并创建一个 ./venv 目录来存放它：

python3 -m venv --system-site-packages ./venv

使用特定于 shell 的命令激活该虚拟环境： (one of three)

source ./venv/bin/activate  # sh, bash, or zsh
. ./venv/bin/activate.fish  # fish
source ./venv/bin/activate.csh  # csh or tcsh

当虚拟环境处于有效状态时，shell 提示符带有 (venv) 前缀。
在不影响主机系统设置的情况下，在虚拟环境中安装软件包。首先升级 pip

pip install --upgrade pip
pip list  # show packages installed within the virtual environment


虚拟环境安装 tensorflow
pip install --upgrade tensorflow
验证安装效果：
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

tensorflow version
pip list | grep tensorflow

之后退出虚拟环境：
deactivate  # don't exit until you're done using TensorFlow

