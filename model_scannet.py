import tensorflow as tf
from tensorflow import keras

from pointconv.layers import PointConvSA, PointConvFP


class PointConvModel(keras.Model):

    def __init__(self, batch_size, bn=False, num_classes=21):
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
        
        out_ch = 512

        self.sa_layer1 = PointConvSA(
            npoint=1024, radius=0.1, sigma=self.sigma, K=self.K, mlp=[32, 32, 64], bn=self.bn)
        self.sa_layer2 = PointConvSA(
            npoint=256, radius=0.2, sigma=2*self.sigma, K=self.K, mlp=[64, 64, 128], bn=self.bn)
        self.sa_layer3 = PointConvSA(
            npoint=64, radius=0.4, sigma=4*self.sigma, K=self.K, mlp=[128, 128, 256], bn=self.bn)
        self.sa_layer4 = PointConvSA(
            npoint=36, radius=0.8, sigma=8*self.sigma, K=self.K, mlp=[256, 256, 512], bn=self.bn)


        self.fp_layer1 = PointConvFP(
            radius=0.8, sigma=8*self.sigma, K=16, mlp=[out_ch, 512], out_ch=out_ch, bn=self.bn)
        self.fp_layer2 = PointConvFP(
            radius=0.4, sigma=4*self.sigma, K=16, mlp=[256, 256], out_ch=out_ch, bn=self.bn)
        self.fp_layer3 = PointConvFP(
            radius=0.2, sigma=2*self.sigma, K=16, mlp=[256, 128], out_ch=out_ch, bn=self.bn)
        self.fp_layer4 = PointConvFP(
            radius=0.1, sigma=self.sigma, K=16, mlp=[128, 128, 128], out_ch=out_ch, bn=self.bn)

        self.dense1 = keras.layers.Dense(128, activation=self.activation)
        self.dropout1 = keras.layers.Dropout(0.4)
        self.dense2 = keras.layers.Dense(self.num_classes, activation=tf.nn.softmax)

    def forward_pass(self, input, training):

        l0_xyz = input
        l0_points = None

        l1_xyz, l1_points = self.sa_layer1(l0_xyz, l0_points, training=training)
        l2_xyz, l2_points = self.sa_layer2(l1_xyz, l1_points, training=training)
        l3_xyz, l3_points = self.sa_layer3(l2_xyz, l2_points, training=training)
        l4_xyz, l4_points = self.sa_layer4(l3_xyz, l3_points, training=training)

        l3_points = self.fp_layer1(l3_xyz, l4_xyz, l3_points, l4_points, training=training)
        l2_points = self.fp_layer2(l2_xyz, l3_xyz, l2_points, l3_points, training=training)
        l1_points = self.fp_layer3(l1_xyz, l2_xyz, l1_points, l2_points, training=training)
        points = self.fp_layer4(l0_xyz, l1_xyz, l0_points, l1_points)

        net = self.dense1(points)
        net = self.dropout1(net)

        pred = self.dense2(net)

        return pred

    def train_step(self, input):

        with tf.GradientTape() as tape:

            pred = self.forward_pass(input[0], True)
            loss = self.compiled_loss(input[1], pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(input[1], pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, input):

        pred = self.forward_pass(input[0], False)
        loss = self.compiled_loss(input[1], pred)

        self.compiled_metrics.update_state(input[1], pred)

        return {m.name: m.result() for m in self.metrics}

    def call(self, input, training=False):

        return self.forward_pass(input, training)
