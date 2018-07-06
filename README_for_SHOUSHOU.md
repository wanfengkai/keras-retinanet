# Keras RetinaNet for shoushou


## Enter your anaconda 

## Installation

1) Clone this repository, `git clone https://github.com/wanfengkai/keras-retinanet.git`

2) In the repository, execute `pip install . --user`.

3) 设置基本环境，只要`python setup.py build_ext --inplace`

## Train your model

1） 数据准备：将data_to_text.py生成的文件放到`csv_files_for_train/`下面并且重命名成`annotation.txt`

在 代码所在路径下执行以下代码：
`keras_retinanet/bin/train.py csv ./csv_files_for_train/annotation.txt ./csv_files_for_train/ID_mapping.txt`


## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb).
In general, inference of the network works as follows:
```python
boxes, scores, labels = model.predict_on_batch(inputs)
```

Where `boxes` are shaped `(None, None, 4)` (for `(x1, y1, x2, y2)`), scores is shaped `(None, None)` (classification score) and labels is shaped `(None, None)` (label corresponding to the score). In all three outputs, the first dimension represents the shape and the second dimension indexes the list of detections.

Loading models can be done in the following manner:
```python
from keras_retinanet.models import load_model
model = load_model('/path/to/model.h5', backbone_name='resnet50')
```

Execution time on NVIDIA Pascal Titan X is roughly 75msec for an image of shape `1000x800x3`.

### Converting a training model to inference model
The training procedure of `keras-retinanet` works with *training models*. These are stripped down versions compared to the *inference model* and only contains the layers necessary for training (regression and classification values). If you wish to do inference on a model (perform object detection on an image), you need to convert the trained model to an inference model. This is done as follows:

```shell
# Running directly from the repository:
keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5

# Using the installed script:
retinanet-convert-model /path/to/training/model.h5 /path/to/save/inference/model.h5
```

Most scripts (like `retinanet-evaluate`) also support converting on the fly, using the `--convert-model` argument.

