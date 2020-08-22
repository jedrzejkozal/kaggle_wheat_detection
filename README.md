# kaggle wheat detection
Code for kaggle competition wheat detection. Created mostly to play around with object detection models and use AWS for neural network training. 

Model used was torchvision implementation of Faster RCNN. 

Used technologies:
 - pytorch
 - torchvision
 - albumentations
 - tensorboard

## Setup

I tired to automate whole setup, so starting training on new AWS machine should be easy. There are 3 considerations I had in mind:

 - data
 - environment
 - training resutls
 
Data and environment setup are described below. 
Hyperparameters are described in Training artifacts section. 
One remark: after training commit results of training to repo before terminating AWS machine (this is inconvinience that should be removed).

### Data

Kaggle dataset is not included in this repo. You must download it separatly. It can be done with kaggle API:

```
kaggle competitions download -c global-wheat-detection
```

Put it into data/global-wheat-detection.zip in this repo and run setup.sh. This script will uznip dataset, split to train-val and move data to correct location. After that you should be able to training.

### Environment

Create conda env using environment.yml file in root directory. It has a lot of dependencies it can take a while.

## Training

To start training run:

```
python torchvision_training.py
```

You can specify data directories via comandline (but if setup.sh script from previous paragraph was used default data paths should be fine). 
Hyperparameters for training can also be secified using commandline.

### Training artifacts

All hyperparameters and metrics from training are logged using tensorboard. Results will be stored into runs/ directory. To run tensorboard ui execute:

```
tensorboard --logdir=runs
```

By default tensorboard will start at http://localhost:6006/.


After training model will be saved into faster_rcnn.pt file. Predictions for test set will be generated using saved model and stored into submission.csv.
