"""
Helper Function for PointConv
Author: Wenxuan Wu
Date: July 2018
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import KDTree

from .cpp_modules import (
	farthest_point_sample,
	gather_point,
	query_ball_point,
	group_point,
    three_nn
)

def knn_kdtree(nsample, xyz, new_xyz):
    batch_size = xyz.shape[0]
    n_points = new_xyz.shape[1]

    indices = np.zeros((batch_size, n_points, nsample), dtype=np.int32)
    for batch_idx in range(batch_size):
        X = xyz.numpy()[batch_idx, ...]
        q_X = new_xyz[batch_idx, ...]
        kdt = KDTree(X, leaf_size=30)
        _, indices[batch_idx] = kdt.query(q_X, k=nsample)

    return indices


def kernel_density_estimation_ball(pts, radius, sigma, N_points=128, is_norm=False):

    idx, pts_cnt = query_ball_point(radius, N_points, pts, pts)
    g_pts = group_point(pts, idx)
    g_pts -= tf.tile(tf.expand_dims(pts, 2), [1, 1, N_points, 1])

    R = tf.sqrt(sigma)
    xRinv = tf.math.divide(g_pts, R)
    quadform = tf.reduce_sum(tf.square(xRinv), axis=-1)
    logsqrtdetSigma = tf.math.log(R) * 3
    mvnpdf = tf.exp(-0.5 * quadform - logsqrtdetSigma -
                    3 * tf.math.log(2 * 3.1415926) / 2)

    first_val, _ = tf.split(mvnpdf, [1, N_points - 1], axis=2)

    mvnpdf = tf.reduce_sum(mvnpdf, axis=2, keepdims=True)

    num_val_to_sub = tf.expand_dims(
        tf.cast(tf.subtract(N_points, pts_cnt), dtype=tf.float32), axis=-1)

    val_to_sub = tf.multiply(first_val, num_val_to_sub)

    mvnpdf = tf.subtract(mvnpdf, val_to_sub)

    scale = tf.math.divide(1.0, tf.expand_dims(
        tf.cast(pts_cnt, dtype=tf.float32), axis=-1))
    density = tf.multiply(mvnpdf, scale)

    if is_norm:
        density_max = tf.reduce_max(density, axis=1, keepdims=True)
        density = tf.math.divide(density, density_max)

    return density


def kernel_density_estimation(pts, sigma, kpoint=32, is_norm=False):
    with tf.variable_scope("ComputeDensity") as sc:
        batch_size = pts.get_shape()[0]
        num_points = pts.get_shape()[1]
        if num_points < kpoint:
            kpoint = num_points.value - 1
        with tf.device('/cpu:0'):
            point_indices = tf.py_function(
                knn_kdtree, [kpoint, pts, pts], tf.int32)
        batch_indices = tf.tile(tf.reshape(
            tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, kpoint, 1))
        idx = tf.concat(
            [batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
        idx.set_shape([batch_size, num_points, kpoint, 2])

        grouped_pts = tf.gather_nd(pts, idx)
        # translation normalization
        grouped_pts -= tf.tile(tf.expand_dims(pts, 2), [1, 1, kpoint, 1])

        R = tf.sqrt(sigma)
        xRinv = tf.div(grouped_pts, R)
        quadform = tf.reduce_sum(tf.square(xRinv), axis=-1)
        logsqrtdetSigma = tf.log(R) * 3
        mvnpdf = tf.exp(-0.5 * quadform - logsqrtdetSigma -
                        3 * tf.log(2 * 3.1415926) / 2)
        mvnpdf = tf.reduce_sum(mvnpdf, axis=2, keepdims=True)

        scale = 1.0 / kpoint
        density = tf.multiply(mvnpdf, scale)

        if is_norm:
            density_max = tf.reduce_max(density, axis=1, keepdims=True)
            density = tf.div(density, density_max)

        return density


def sampling(npoint, pts):
    '''
    inputs:
    npoint: scalar, number of points to sample
    pointcloud: B * N * 3, input point cloud
    output:
    sub_pts: B * npoint * 3, sub-sampled point cloud
    '''

    sub_pts = gather_point(
        pts, farthest_point_sample(npoint, pts))
    return sub_pts


def grouping(feature, K, src_xyz, q_xyz, use_xyz=True):
    '''
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    '''

    batch_size = src_xyz.get_shape()[0]
    npoint = q_xyz.get_shape()[1]

    point_indices = tf.py_function(knn_kdtree, [K, src_xyz, q_xyz], tf.int32)
    batch_indices = tf.tile(tf.reshape(
        tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
    idx = tf.concat(
        [batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
    idx.set_shape([batch_size, npoint, K, 2])

    grouped_xyz = tf.gather_nd(src_xyz, idx)
    # translation normalization
    grouped_xyz -= tf.tile(tf.expand_dims(q_xyz, 2), [1, 1, K, 1])

    grouped_feature = tf.gather_nd(feature, idx)
    if use_xyz:
        new_points = tf.concat([grouped_xyz, grouped_feature], axis=-1)
    else:
        new_points = grouped_feature

    return grouped_xyz, new_points, idx


def grouping_all(feature, src_xyz, use_xyz=True):

    batch_size = src_xyz.get_shape()[0]
    npoint = src_xyz.get_shape()[1]

    new_xyz = tf.reduce_mean(src_xyz, axis=1, keepdims=True)
    new_xyz = tf.reshape(src_xyz, (batch_size, 1, src_xyz.shape[1], 3)) - tf.reshape(new_xyz, (batch_size, 1, 1, 3))

    idx = tf.constant(np.tile(np.array(range(npoint)).reshape((1, 1, npoint, 1)), (batch_size, 1, 1, 1)), tf.int32)
    idx = tf.concat([tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), [1, 1, new_xyz.shape[2], 1]), idx], -1)

    grouped_xyz = tf.reshape(src_xyz, (batch_size, 1, npoint, 3))

    if feature is not None:
        if use_xyz:
            new_points = tf.concat([src_xyz, feature], axis=2)
        else:
            new_points = feature
        new_points = tf.expand_dims(new_points, 1)
    else:
        new_points = grouped_xyz

    return grouped_xyz, new_points, idx


class Conv2d(keras.layers.Layer):

    def __init__(self, filters, strides=[1, 1], activation=tf.nn.relu, padding='VALID', initializer='glorot_normal', bn=False):
        super(Conv2d, self).__init__()

        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.initializer = initializer
        self.bn = bn

    def build(self, input_shape):

        self.w = self.add_weight(
            shape=(1, 1, input_shape[-1], self.filters),
            initializer=self.initializer,
            trainable=True,
            name='pnet_conv'
        )

        if self.bn:
            self.bn_layer = keras.layers.BatchNormalization()

        super(Conv2d, self).build(input_shape)

    def call(self, inputs, training=True):

        points = tf.nn.conv2d(inputs, filters=self.w, strides=self.strides, padding=self.padding)
        if self.bn:points = self.bn_layer(points, training=training)
        if self.activation:points = self.activation(points)

        return points
