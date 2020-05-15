import os
import sys

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model_scannet import PointConvModel
from tensorflow import keras
import tensorflow as tf

tf.random.set_seed(1234)


def load_dataset(in_file, batch_size):

    assert os.path.isfile(in_file), '[error] dataset path not found'

    n_points = 8192
    shuffle_buffer = 1000

    def _extract_fn(data_record):

        in_features = {
            'points': tf.io.FixedLenFeature([n_points * 3], tf.float32),
            'labels': tf.io.FixedLenFeature([n_points], tf.int64)
        }

        return tf.io.parse_single_example(data_record, in_features)

    def _preprocess_fn(sample):

        points = sample['points']
        labels = sample['labels']

        points = tf.reshape(points, (n_points, 3))
        labels = tf.reshape(labels, (n_points, 1))

        shuffle_idx = tf.range(points.shape[0])
        points = tf.gather(points, shuffle_idx)
        labels = tf.gather(labels, shuffle_idx)

        return points, labels

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
        validation_data=val_ds,
        validation_steps=10,
        validation_freq=1,
        callbacks=callbacks,
        epochs=100,
        verbose=1
    )


if __name__ == '__main__':

    config = {
        'train_ds': './data/scannet_train.tfrecord',
        'val_ds': './data/scannet_val.tfrecord',
        'batch_size': 4,
        'lr': 1e-3,
        'bn': False,
        'log_dir': 'scannet_2'
    }

    train()
