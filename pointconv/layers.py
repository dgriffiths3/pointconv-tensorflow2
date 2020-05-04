import sys
sys.path.insert(0, './')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, BatchNormalization

from pointconv import utils

class PointConvSA(keras.layers.Layer):

    def __init__(self, npoint, radius, sigma, K, mlp, group_all=False, activation=tf.nn.relu, bn=False):

        super(PointConvSA, self).__init__()

        self.npoint = npoint
        self.radius = radius
        self.sigma = sigma
        self.K = K
        self.mlp = mlp
        self.group_all = group_all
        self.activation = activation
        self.bn = bn

        self.mlp_list = []
        self.weightnet_hidden = []
        self.nonlinear_transform = []


    def build(self, input_shape):

        for i, n_filters in enumerate(self.mlp):
            self.mlp_list.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))

        for i, n_filters in enumerate([32]):
            self.weightnet_hidden.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))

        for i, n_filters in enumerate([16, 1]):
            self.nonlinear_transform.append(utils.Conv2d(n_filters, activation=self.activation, bn=self.bn))

        self.np_conv = utils.Conv2d(self.mlp[-1], strides=[1, self.mlp[-1]], activation=self.activation, bn=self.bn)

        super(PointConvSA, self).build(input_shape)


    def call(self, xyz, feature, training=True):

        num_points = xyz.get_shape()[1]

        if feature is None: 
            feature = tf.identity(xyz)

        if num_points == self.npoint:
            new_xyz = xyz
        else:
            new_xyz = utils.sampling(self.npoint, xyz)

        if self.group_all == True:
            grouped_xyz, grouped_feature, idx = utils.grouping_all(feature, xyz)
        else:
            grouped_xyz, grouped_feature, idx = utils.grouping(feature, self.K, xyz, new_xyz)

        density = utils.kernel_density_estimation_ball(xyz, self.radius, self.sigma)
        inverse_density = tf.math.divide(1.0, density)
        grouped_density = tf.gather_nd(inverse_density, idx) # (batch_size, npoint, nsample, 1)
        inverse_max_density = tf.reduce_max(grouped_density, axis = 2, keepdims = True)
        density_scale = tf.math.divide(grouped_density, inverse_max_density)

        for i, mlp_layer in enumerate(self.mlp_list):
            grouped_feature = mlp_layer(grouped_feature, training=training)

        for i, mlp_layer in enumerate(self.weightnet_hidden):
            weight = mlp_layer(grouped_xyz, training=training)

        for i, mlp_layer in enumerate(self.nonlinear_transform):
            density_scale = mlp_layer(density_scale, training=training)

        new_points = tf.math.multiply(grouped_feature, density_scale)
        new_points = tf.transpose(new_points, [0, 1, 3, 2])
        new_points = tf.linalg.matmul(new_points, weight)

        new_points = self.np_conv(new_points, training=training)
        new_points = tf.squeeze(new_points, [2])

        return new_xyz, new_points
