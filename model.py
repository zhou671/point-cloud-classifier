import tensorflow as tf
import numpy as np

NUM_POINT = 1024

class PointNet(tf.keras.Model):
    def __init__(self, probs):
        
        super(PointNet, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam()
        self.trans1 = Trans_Net(3)
        self.conv1 = Conv2d(64)
        self.conv2 = Conv2d(64)
        self.trans2 = Trans_Net(64)
        self.conv3 = Conv2d(64)
        self.conv4 = Conv2d(128)
        self.conv5 = Conv2d(1024)
        self.dense1 = Dense(512)
        self.dense2 = Dense(256)
        self.dense3 = tf.keras.layers.Dense(2)
        self.probs = probs
        
    
    @tf.function
    def call(self, point_clouds):
        # when probs == 0.0, dropout turns off, e.g. when testing
        batch_size = point_clouds.get_shape()[0]
        point_clouds = tf.cast(point_clouds, tf.float32)
        
        transform = self.trans1(point_clouds)
        net = tf.matmul(point_clouds, transform)
        net = tf.reshape(net, [batch_size, NUM_POINT, 1, -1])
        net = self.conv1(net)
        net = self.conv2(net)
        
        transform = self.trans2(net)
        net = tf.reshape(net, [batch_size, NUM_POINT, -1])
        net = tf.matmul(net, transform)
        net = tf.reshape(net, [batch_size, NUM_POINT, 1, -1])
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.conv5(net)
        
        net = tf.nn.max_pool2d(net, ksize = [NUM_POINT, 1], strides = [1, 1], padding = "VALID")
        net = tf.reshape(net, [batch_size, -1])
        
        net = self.dense1(net)
        net = tf.nn.dropout(net, self.probs)
        net = self.dense2(net)
        net = tf.nn.dropout(net, self.probs)
        net = self.dense3(net)
        
        return net
    
    def loss(self, pred, labels):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels)
        value = tf.reduce_mean(loss)
        argmax = tf.cast(tf.math.argmax(pred, axis = 1), dtype = tf.int64)
        acc = tf.reduce_mean(tf.cast(tf.equal(argmax, labels), dtype = tf.float32))
        return value, acc




class Trans_Net(tf.keras.layers.Layer):
    def __init__(self, K):
        super(Trans_Net, self).__init__()
        self.K = K
        self.conv1 = Conv2d(64)
        self.conv2 = Conv2d(128)
        self.conv3 = Conv2d(1024)
        self.dense1 = Dense(512)
        self.dense2 = Dense(256)
        
        self.to_trans = tf.keras.layers.Dense(K * K)
        
    @tf.function
    def call(self, inputs):
        # input should be shape [batch_size, NUM_POINT, 1, num_channel]
        
        batch_size = inputs.get_shape()[0]
        net = tf.reshape(inputs, [batch_size, NUM_POINT, 1, -1])
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.conv3(net)
        net = tf.nn.max_pool2d(net, ksize = [NUM_POINT, 1], strides = [1, 1], padding = "VALID")
        net = tf.reshape(net, [batch_size, -1])
        net = self.dense1(net)
        net = self.dense2(net)
        net = self.to_trans(net)
        eye = np.eye(self.K).flatten()
        return tf.reshape(net + eye, [batch_size, self.K, self.K])


class Conv2d(tf.keras.layers.Layer):
    def __init__(self, num_channel):
        super(Conv2d, self).__init__()
        self.conv = tf.keras.layers.Conv2D(num_channel, [1, 1], strides=[1,1], padding='valid')
        self.bn = tf.keras.layers.BatchNormalization()
        
    @tf.function
    def call(self, inputs):
        # input should be shape [batch_size, NUM_POINT, 1, num_channel]
        inputs = self.conv(inputs)
        inputs = self.bn(inputs)
        inputs = tf.nn.relu(inputs)
        return inputs
    
class Dense(tf.keras.layers.Layer):
    def __init__(self, num_channel):
        super(Dense, self).__init__()
        self.dense = tf.keras.layers.Dense(num_channel)
        self.bn = tf.keras.layers.BatchNormalization()
        
    @tf.function
    def call(self, inputs):
        # input should be shape [batch_size, NUM_POINT, 1, num_channel]
        inputs = self.dense(inputs)
        inputs = self.bn(inputs)
        inputs = tf.nn.relu(inputs)
        return inputs
