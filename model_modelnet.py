import tensorflow as tf
from tensorflow import keras

from pointconv.layers import PointConvSA


class PointConvModel(keras.Model):

    def __init__(self, batch_size, bn=False, num_classes=40):
        super(PointConvModel, self).__init__()

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.activation = tf.nn.relu
        self.kernel_initializer = 'glorot_normal'
        self.sigma = 0.1
        self.K = 32
        self.bn = bn

        self.init_network()


    def init_network(self):

        self.layer1 = PointConvSA(
            npoint = 512,
            radius = 0.1,
            sigma = self.sigma,
            K = self.K,
            mlp = [64, 64, 128],
            bn = self.bn
        )

        self.layer2 = PointConvSA(
            npoint = 128,
            radius = 0.2,
            sigma = 2*self.sigma,
            K = self.K,
            mlp = [128, 128, 256],
            bn = self.bn
        )

        self.layer3 = PointConvSA(
            npoint = 1,
            radius = 0.8,
            sigma = 4*self.sigma,
            K = self.K,
            mlp = [256, 512, 1024],
            group_all=True,
            bn = self.bn
        )

        self.dense1 = keras.layers.Dense(512, activation=self.activation)
        self.dropout1 = keras.layers.Dropout(0.4)

        self.dense2 = keras.layers.Dense(256, activation=self.activation)
        self.dropout2 = keras.layers.Dropout(0.4)

        self.dense3 = keras.layers.Dense(self.num_classes, activation=tf.nn.softmax)


    def forward_pass(self, input, training):

        xyz, points = self.layer1(input, None, training=training)
        xyz, points = self.layer2(xyz, points, training=training)
        xyz, points = self.layer3(xyz, points, training=training)

        net = tf.reshape(points, (self.batch_size, -1))

        net = self.dense1(net)
        net = self.dropout1(net)

        net = self.dense2(net)
        net = self.dropout2(net)
        
        pred = self.dense3(net)

        return pred


    def train_step(self, input):

        with tf.GradientTape() as tape:

            pred = self.forward_pass(input[0], True)
            loss = self.compiled_loss(input[1], pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(input[1], pred)

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, input):

        pred = self.forward_pass(input[0], False)
        loss = self.compiled_loss(input[1], pred)

        self.compiled_metrics.update_state(input[1], pred)

        return {m.name: m.result() for m in self.metrics}


    def call(self, input, training=False):

        return self.forward_pass(input, training)
