import tensorflow as tf
from nets.vgg19 import VGG19
from nets.vgg16 import VGG16
import utils
import numpy as np


def vgg19_test():
    vgg19 = VGG19('data/checkpoints/vgg_19.ckpt')
    inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg19.build_model(inputs, is_train=False)

    test_img_path = 'data/test_imgs/ILSVRC2012_val_00000003.JPEG'
    test_img = utils.read_img(test_img_path, expand_dims=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(vgg19.fc8, feed_dict={inputs: test_img})
    prob_lable = int(np.argmax(output, axis=1))
    print(prob_lable, utils.get_class_name_by_id(prob_lable))


def vgg16_test():
    vgg16 = VGG16('data/checkpoints/vgg_16.ckpt')
    inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg16.build_model(inputs, is_train=False)

    test_img_path = 'data/test_imgs/ILSVRC2012_val_00000003.JPEG'
    test_img = utils.read_img(test_img_path, expand_dims=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(vgg16.fc8, feed_dict={inputs: test_img})
    prob_lable = int(np.argmax(output, axis=1))
    print(prob_lable, utils.get_class_name_by_id(prob_lable))



if __name__ == "__main__":
    # vgg19_test()
    vgg16_test()
