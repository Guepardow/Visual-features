# An Empirical Analysis of Visual Features for Multiple Object Tracking in Urban Scenes
**Authors : Mehdi Miah, Justine Pépin, Nicolas Saunier & Guillaume-Alexandre Bilodeau** 
**Polytechnique Montréal**

[Project page] [arXiv] [ICPR paper]

<figure class="video_container">
  <video controls="false" allowfullscreen="true" width="480" height="270" autoplay loop muted markdown="1">
    <source src="./doc/visual_features.mp4" type="video/mp4">
  </video>
</figure>

## Objective

Rank visual descriptors for multiple object tracking focused on urban scenes

## Installation

### Requirements

Linux and Windows supported. Python 3.6, Pytorch 1.4, CUDA 10.0

### Clone repositories

``cd src
git clone https://github.com/KaiyangZhou/deep-person-reid.git (torchreid 1.2.7)
mv deep-person-reid/ deep_person_reid/
pip install efficientnet_pytorch
git clone https://github.com/cw1204772/AIC2018_iamai.git
``

### Datasets

You can download the following datasets :
- WildTrack : [https://www.epfl.ch/labs/cvlab/data/data-wildtrack/](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)
- MOT17 : [https://motchallenge.net/data/MOT17/](https://motchallenge.net/data/MOT17/)
- DETRAC : [http://detrac-db.rit.albany.edu/](http://detrac-db.rit.albany.edu/)
- UAVDT : [https://sites.google.com/site/daviddo0323/projects/uavdt](https://sites.google.com/site/daviddo0323/projects/uavdt)

Change the path to data in the file ``./src/dataset.py``.

### Weights

The weights for VGG-19, ResNet-18, DenseNet-121 come from Pytorch.

The weights for Efficient-B0 come from ``efficientnet_pytorch`` (automatically downloaded).

The weights for pedestrian ReID come from [https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html#market1501-dukemtmc-reid](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html#market1501-dukemtmc-reid)

The weights for vehicles ReID come from [https://github.com/cw1204772/AIC2018_iamai#demo](https://github.com/cw1204772/AIC2018_iamai#demo)


### Final structure of files

```bash
.
+-- doc
+-- results
|   +-- figure
|   +-- logs
+-- src
|   +-- affinity.py
|   +-- analysis_rank.py
|   +-- analysis_size.py
|   +-- appearances.py
|   +-- dataset.py
|   +-- main.py
|   +-- utils.py
|   +-- AIC2018_iamai
|   +-- deep_person_reid
+-- weights
|   +-- model_880_base.ckpt
|   +-- osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
```

## Run experiments

``
python main.py --dataset="DETRAC" --scene="20011" --features="resnet18" --sigma=10
``

``
python analysis_rank.py --dataset="DETRAC"
``

``
python analysis_size.py --dataset="DETRAC"
``

## Results

## Citation and acknowledgment

```bibtex
@inproceedings{miah2020empirical,
    title = {An {Empirical} {Analysis} of {Visual} {Features} for {Multiple} {Object} {Tracking} in {Urban} {Scenes}},
    author = {Miah, Mehdi and Pépin, Justine and Saunier, Nicolas and Bilodeau, Guillaume-Alexandre},
    booktitle = {International {Conference} on {Pattern} {Recognition} ({ICPR})},
    year = {2020}
}
```

We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), [CRDPJ 528786 - 18], [DG 2017-06115] and the support of Arcturus Networks.
