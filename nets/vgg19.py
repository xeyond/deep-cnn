import tensorflow as tf

RGB_MEAN = [123.68, 116.78, 103.94]


class VGG19(object):
    def __init__(self, weights_path=None, n_class=1000):
        self.weights_dict = self._parse_ckpt(weights_path)
        self.n_class = n_class

    def _parse_ckpt(self, weights_path):
        weights_dict = dict()
        if not weights_path:
            return weights_dict
        reader = tf.train.NewCheckpointReader(weights_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for var_name in sorted(var_to_shape_map):
            if 'global_step' in var_name:
                continue
            new_var_name = "%s/%s" % tuple(var_name.split('/')[-2:])
            weights_dict[new_var_name] = reader.get_tensor(var_name)
        return weights_dict

    def _get_var(self, name, shape):
        if name in self.weights_dict:
            var_initializer = tf.initializers.constant(self.weights_dict[name])
        else:
            var_initializer = tf.initializers.truncated_normal(shape, mean=0.0, stddev=0.001)
        return tf.get_variable(name=name, shape=shape, initializer=var_initializer)

    def _conv2d(self, name, inputs, filters, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None):
        input_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size[0], kernel_size[1], input_channels, filters]
        kernel_weights = self._get_var(name + '/weights', shape=kernel_shape)
        kernel_biases = self._get_var(name + '/biases', shape=[filters])
        conv = tf.nn.conv2d(inputs, filter=kernel_weights, strides=[1, strides[0], strides[1], 1], padding=padding,
                            name=name) + kernel_biases
        if activation:
            conv = activation(conv)
        return conv

    def build_model(self, inputs, is_train=True, keep_prob=0.5):
        inputs = inputs - RGB_MEAN
        self.conv1_1 = self._conv2d('conv1_1', inputs, filters=64, activation=tf.nn.relu)
        self.conv1_2 = self._conv2d('conv1_2', self.conv1_1, filters=64, activation=tf.nn.relu)
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv2_1 = self._conv2d('conv2_1', self.pool1, filters=128, activation=tf.nn.relu)
        self.conv2_2 = self._conv2d('conv2_2', self.conv2_1, filters=128, activation=tf.nn.relu)
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv3_1 = self._conv2d('conv3_1', self.pool2, filters=256, activation=tf.nn.relu)
        self.conv3_2 = self._conv2d('conv3_2', self.conv3_1, filters=256, activation=tf.nn.relu)
        self.conv3_3 = self._conv2d('conv3_3', self.conv3_2, filters=256, activation=tf.nn.relu)
        self.conv3_4 = self._conv2d('conv3_4', self.conv3_3, filters=256, activation=tf.nn.relu)
        self.pool3 = tf.nn.max_pool(self.conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv4_1 = self._conv2d('conv4_1', self.pool3, filters=512, activation=tf.nn.relu)
        self.conv4_2 = self._conv2d('conv4_2', self.conv4_1, filters=512, activation=tf.nn.relu)
        self.conv4_3 = self._conv2d('conv4_3', self.conv4_2, filters=512, activation=tf.nn.relu)
        self.conv4_4 = self._conv2d('conv4_4', self.conv4_3, filters=512, activation=tf.nn.relu)
        self.pool4 = tf.nn.max_pool(self.conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv5_1 = self._conv2d('conv5_1', self.pool4, filters=512, activation=tf.nn.relu)
        self.conv5_2 = self._conv2d('conv5_2', self.conv5_1, filters=512, activation=tf.nn.relu)
        self.conv5_3 = self._conv2d('conv5_3', self.conv5_2, filters=512, activation=tf.nn.relu)
        self.conv5_4 = self._conv2d('conv5_4', self.conv5_3, filters=512, activation=tf.nn.relu)
        self.pool5 = tf.nn.max_pool(self.conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.fc6 = self._conv2d('fc6', self.pool5, filters=4096, kernel_size=[7, 7], padding='VALID',
                                activation=tf.nn.relu)
        if is_train:
            self.fc6_drop = tf.nn.dropout(self.fc6, keep_prob=keep_prob)
        else:
            self.fc6_drop = self.fc6

        self.fc7 = self._conv2d('fc7', self.fc6_drop, filters=4096, kernel_size=[1, 1], padding='VALID',
                                activation=tf.nn.relu)
        if is_train:
            self.fc7_drop = tf.nn.dropout(self.fc7, keep_prob=keep_prob)
        else:
            self.fc7_drop = self.fc7

        self.fc8 = self._conv2d('fc8', self.fc7_drop, filters=self.n_class, kernel_size=[1, 1], padding='VALID',
                                activation=tf.nn.softmax)
        self.fc8 = tf.squeeze(self.fc8, axis=[1, 2])
        print(self.fc8)