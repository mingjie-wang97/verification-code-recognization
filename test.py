import tensorflow.compat.v1 as tf
tf.disable_v2_behavior
from train import cnn_graph
from process import vec2text,convert2gray,wrap_gen_captcha_text_and_image
from getimg import CAPTCHA_HEIGHT, CAPTCHA_WIDTH, CAPTCHA_LEN, CAPTCHA_LIST

import numpy as np
import random

# 验证码图片转化为文本
def captcha2text(image_list, height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH):
    # disable the tensor flow 2 eager execution function
    tf.compat.v1.disable_eager_execution()

    x = tf.placeholder(tf.float32, [None, height * width])
    keep_prob = tf.placeholder(tf.float32)
    y_conv = cnn_graph(x, keep_prob, (height, width))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        print(tf.train.latest_checkpoint('.'))
        predict = tf.argmax(tf.reshape(y_conv, [-1, CAPTCHA_LEN, len(CAPTCHA_LIST)]), 2)
        vector_list = sess.run(predict, feed_dict={x: image_list, keep_prob: 1})
        vector_list = vector_list.tolist()

        text_list = [vec2text(vector) for vector in vector_list]

        return text_list

if __name__ == '__main__':
    print("application start running.....")
    text, image = wrap_gen_captcha_text_and_image()
    text_list = []
    image_list = []
    for _ in range(1000):
        text_list.append(random.choice(text))
        image_a = image[text.index(text_list[-1])]
        img_array = np.array(image_a)
        cur_image = convert2gray(img_array)
        cur_image = cur_image.flatten() / 255
        image_list.append(cur_image)
    pre_text = captcha2text(image_list)
    sucess = 0
    for _ in range(1000):
        if pre_text[_] == text_list[_]:
            sucess += 1
        #     print(' 正确验证码:', text_list[_], "识别出来的：", pre_text[_],"  TURE")
        # else:
        #     print(' 正确验证码:', text_list[_], "识别出来的：", pre_text[_], "FLASE")
    print("Overall correctness of ", sucess/10, "%")