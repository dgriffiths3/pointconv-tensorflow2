# PointConv tensorflow 2.0 layers

This repository containts implementations of the PointConv (Wu et al, 2019) feature encoder and feature decoder layers as `tf.keras.layers` classes. This allows for PointConv layers to be used as part of the standard `tf.keras` api. The repository does not aim to be an exact implementation of the original repostiroy, rather a useful tool for building custom models or simple backend encoders for unordered point sets. For more details regarding the technical details check out the [original paper](https://arxiv.org/abs/1811.07246) and [github page](https://github.com/DylanWusee/pointconv). The implementation also matches the style of the [PointNet++ keras layers](https://github.com/dgriffiths3/pointnet2-tensorflow2).

> Note: I have only implemented the feature encoding layer. I will add the decoder as soon as I find time. Once the decoder is done, I will upload a ScanNet per-point segmentation model example as well.

## Setup

Requirements:

```
python >= 3.6
tensorflow >= 2.2+
cuda == 10.1
```
> Note: This repository uses the `train_step` model override which is new for `tensorflow 2.2.0`, as such if you wish to use the provided training scripts it is important your tensorflow is not an older version. The layers will work for tensorflow 2.0+.


To compile the C++ tensorflow ops, first ensure the `CUDA_ROOT` path in `tf_ops/compile_ops.sh` points correctly to your cuda folder and then compile the ops with:

```
chmod u+x tf_ops/compile_ops.sh
tf_ops/compile_ops.sh
```

## Usage

The layers follow the standard `tf.keras.layers` api. To import in your own project, copy the `pointconv` and `tf_ops` folders and set a relative path to find the layers. Here is an example of how a simple PointConv SetAbstraction model can be built using `tf.keras.Model()`.

```
from tensorflow import keras
from pointconv.layers import PointConvSetAbstraction

class MyModel(keras.Model):

  def __init__(self, batch_size):
    super(MyModel, self).__init__()

        self.layer1 = PointConvSA(npoint=512, radius=0.1, sigma=0.1, K=32, mlp=[64, 64, 128], bn=True)
        self.layer2 = PointConvSA(npoint=128, radius=0.2, sigma=0.2, K=32, mlp=[128, 128, 256], bn=True)
        self.layer2 = PointConvSA(npoint=1, radius=0.8, sigma=0.4, K=32, mlp=[256, 512, 1024], group_all=True bn=True)

        # To make a classifier, just add some fully-connected layers

        self.fn1 = keras.layers.Dense(512)
        self.fn2 = keras.layers.Dense(256)
        self.fn3 = keras.layers.Dense(n_classes, tf.nn.softmax)
    
  def call(input):

    xyz, points = self.layer1(input, None, training=training)
    xyz, points = self.layer2(xyz, points, training=training)
    xyz, points = self.layer3(xyz, points, training=training)

    net = tf.reshape(points, (self.batch_size, -1))

    net = self.dense1(net)
    net = self.dense2(net)
    pred = self.dense3(net)

    return pred
```

A full working example of an implemented model for classification and point-wise semantic segmentation can be found in `model_modelnet.py` and `model_scannet.py` respectively. To run, first download the training data from [here](https://drive.google.com/drive/folders/1v5B68RHgDI95KM4EhDrRJxLacJAHcoxz) and place in a folder called `data`. Configure the `config` dictionary to point to where you have saved it. Once the `config` is set, start the training with:

```
python train_modelnet.py
```

or:

```
python train_scannet.py
```

If the config is left to the default you can view training logs with:

```
cd <project root>
tensorboard --logdir=logs --port=6006
```
and navigate to `localhost:6006` in a web browser.

## Note

If you use these layers in your project remember to cite the original authors:

```
@article{wu2018pointconv,
  title={PointConv: Deep Convolutional Networks on 3D Point Clouds},
  author={Wu, Wenxuan and Qi, Zhongang and Fuxin, Li},
  journal={arXiv preprint arXiv:1811.07246},
  year={2018}
}
```
