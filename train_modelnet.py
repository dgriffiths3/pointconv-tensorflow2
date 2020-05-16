import os
import sys

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from model_modelnet import PointConvModel

tf.random.set_seed(1234)


def load_dataset(in_file, batch_size):

    assert os.path.isfile(in_file), '[error] dataset path not found'

    n_points = 8192
    shuffle_buffer = 1000

    def _extract_fn(data_record):

        in_features = {
            'points': tf.io.FixedLenFeature([n_points * 3], tf.float32),
            'label': tf.io.FixedLenFeature([1], tf.int64)
        }

        return tf.io.parse_single_example(data_record, in_features)

    def _preprocess_fn(sample):

        points = sample['points']
        label = sample['label']

        points = tf.reshape(points, (n_points, 3))
        points = tf.random.shuffle(points)

        return points, label

    dataset = tf.data.TFRecordDataset(in_file)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def train():

    model = PointConvModel(config['batch_size'], config['bn'])

    train_ds = load_dataset(config['train_ds'], config['batch_size'])
    val_ds = load_dataset(config['val_ds'], config['batch_size'])

    callbacks = [
        keras.callbacks.EarlyStopping(
            'val_sparse_categorical_accuracy', min_delta=0.1, patience=3),
        keras.callbacks.TensorBoard(
            './logs/{}'.format(config['log_dir']), update_freq=50),
        keras.callbacks.ModelCheckpoint(
            './logs/{}/model/weights'.format(config['log_dir']), 'val_sparse_categorical_accuracy', save_best_only=True)
    ]

    model.build((config['batch_size'], 8192, 3))
    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(config['lr']),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    model.fit(
        train_ds,
        validation_data = val_ds,
        validation_steps = 10,
        validation_freq = 1,
        callbacks = callbacks,
        epochs = 100,
        verbose = 1
    )


if __name__ == '__main__':

    config = {
        'train_ds' : './data/modelnet_train.tfrecord',
        'val_ds' : './data/modelnet_val.tfrecord',
        'batch_size' : 8,
        'lr' : 1e-3,
        'bn' : False,
        'log_dir' : 'modelnet_1'
    }

    train()
