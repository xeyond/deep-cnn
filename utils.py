from scipy import misc
import numpy as np
import json

def read_img(img_path, img_size=(224, 224), expand_dims=False):
    img_mat = misc.imread(img_path, mode='RGB')
    img_height, img_width = img_mat.shape[0:2]
    short_edge = min(img_height, img_width)
    h_s = (img_height - short_edge) // 2
    w_s = (img_width - short_edge) // 2
    croped_img = img_mat[h_s:h_s + short_edge, w_s:w_s + short_edge]
    resized_img = misc.imresize(croped_img, size=img_size)
    if expand_dims:
        resized_img = np.expand_dims(resized_img, axis=0)
    return resized_img


def get_class_name_by_id(class_id):
    class_names = [class_name.strip() for class_name in open('dataset/imagenet_classid_to_name', 'r').readlines()]
    return class_names[class_id]


if __name__ == '__main__':
    name = get_class_name_by_id(65)
    print(name)
    test_img_path = 'data/test_imgs/ILSVRC2012_val_00000028.JPEG'
    img = read_img(test_img_path)
    print(img.shape)
