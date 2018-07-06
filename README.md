# Keras RetinaNet [![Build Status](https://travis-ci.org/fizyr/keras-retinanet.svg?branch=master)](https://travis-ci.org/fizyr/keras-retinanet) [![DOI](https://zenodo.org/badge/100249425.svg)](https://zenodo.org/badge/latestdoi/100249425)

Keras implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Installation

1) Clone this repository.

2) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.


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


## Training

# Using the installed script:
# 兽兽你只要做下面这一行就行。
keras_retinanet/bin/train.py csv ./csv_files_for_train/annotation.txt ./csv_files_for_train/ID_mapping.txt
```

In general, the steps to train on your own datasets are:
1) Create a model by calling for instance `keras_retinanet.models.resnet50_retinanet` and compile it.
   Empirically, the following compile arguments have been found to work well:
```python
model.compile(
    loss={
        'regression'    : keras_retinanet.losses.smooth_l1(),
        'classification': keras_retinanet.losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)
```
2) Create generators for training and testing data (an example is show in [`keras_retinanet.preprocessing.PascalVocGenerator`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/preprocessing/pascal_voc.py)).
3) Use `model.fit_generator` to start training.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.


### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Debugging
Creating your own dataset does not always work out of the box. There is a [`debug.py`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/debug.py) tool to help find the most common mistakes.

Particularly helpful is the `--annotations` flag which displays your annotations on the images from your dataset. Annotations are colored in green when there are anchors available and colored in red when there are no anchors available. If an annotation doesn't have anchors available, it means it won't contribute to training. It is normal for a small amount of annotations to show up in red, but if most or all annotations are red there is cause for concern. The most common issues are that the annotations are too small or too oddly shaped (stretched out).

## Results

### MS COCO

## Status
Example output images using `keras-retinanet` are shown below.

<p align="center">
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco1.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco2.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco3.png" alt="Example result of RetinaNet on MS COCO"/>
</p>

### Projects using keras-retinanet
* [NATO Innovation Challenge](https://medium.com/data-from-the-trenches/object-detection-with-deep-learning-on-aerial-imagery-2465078db8a9). The winning team of the NATO Innovation Challenge used keras-retinanet to detect cars in aerial images ([COWC dataset](https://gdo152.llnl.gov/cowc/)).
* [Microsoft Research for Horovod on Azure](https://blogs.technet.microsoft.com/machinelearning/2018/06/20/how-to-do-distributed-deep-learning-for-object-detection-using-horovod-on-azure/). A research project by Microsoft, using keras-retinanet to distribute training over multiple GPUs using Horovod on Azure.
* [Anno-Mage](https://virajmavani.github.io/saiat/). A tool that helps you annotate images, using input from the keras-retinanet COCO model as suggestions.
* [Telenav.AI](https://github.com/Telenav/Telenav.AI/tree/master/retinanet). For the detection of traffic signs using keras-retinanet.
* [Towards Deep Placental Histology Phenotyping](https://github.com/Nellaker-group/TowardsDeepPhenotyping). This research project uses keras-retinanet for analysing the placenta at a cellular level.
* [4k video example](https://www.youtube.com/watch?v=KYueHEMGRos). This demo shows the use of keras-retinanet on a 4k input video.
* [boring-detector](https://github.com/lexfridman/boring-detector). I suppose not all projects need to solve life's biggest questions. This project detects the "The Boring Company" hats in videos.
* [comet.ml](https://towardsdatascience.com/how-i-monitor-and-track-my-machine-learning-experiments-from-anywhere-described-in-13-tweets-ec3d0870af99). Using keras-retinanet in combination with [comet.ml](https://comet.ml) to interactively inspect and compare experiments.

If you have a project based on `keras-retinanet` and would like to have it published here, shoot me a message on Slack.

### Notes
* This repository requires Keras 2.2.0 or higher.
* This repository is [tested](https://github.com/fizyr/keras-retinanet/blob/master/.travis.yml) using OpenCV 3.4.
* This repository is [tested](https://github.com/fizyr/keras-retinanet/blob/master/.travis.yml) using Python 2.7 and 3.6.

Contributions to this project are welcome.

### Discussions
Feel free to join the `#keras-retinanet` [Keras Slack](https://keras-slack-autojoin.herokuapp.com/) channel for discussions and questions.

## FAQ
* **I get the warning `UserWarning: No training configuration found in save file: the model was not compiled. Compile it manually.`, should I be worried?** This warning can safely be ignored during inference.
* **I get the error `ValueError: not enough values to unpack (expected 3, got 2)` during inference, what to do?**. This is because you are using a train model to do inference. See https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model for more information.
* **How do I do transfer learning?** The easiest solution is to use the `--weights` argument when training. Keras will load models, even if the number of classes don't match (it will simply skip loading of weights when there is a mismatch). Run for example `retinanet-train --weights snapshots/some_coco_model.h5 pascal /path/to/pascal` to transfer weights from a COCO model to a PascalVOC training session. If your dataset is small, you can also use the `--freeze-backbone` argument to freeze the backbone layers.
* **How do I change the number / shape of the anchors?** There is no straightforward way (yet) to do this. Look at https://github.com/fizyr/keras-retinanet/issues/421 for a discussion on what is currently the method to do this.
* **I get a loss of `0`, what is going on?** This mostly happens when none of the anchors "fit" on your objects, because they are most likely too small or elongated. You can verify this using the [debug](https://github.com/fizyr/keras-retinanet#debugging) tool.
* **I have an older model, can I use it after an update of keras-retinanet?** This depends on what has changed. If it is a change that doesn't affect the weights then you can "update" models by creating a new retinanet model, loading your old weights using `model.load_weights(weights_path, by_name=True)` and saving this model. If the change has been too significant, you should retrain your model (you can try to load in the weights from your old model when starting training, this might be a better starting position than ImageNet).
* **I get the error `ModuleNotFoundError: No module named 'keras_retinanet.utils.compute_overlap'`, how do I fix this?** Most likely you are running the code from the cloned repository. This is fine, but you need to compile some extensions for this to work (`python setup.py build_ext --inplace`).
