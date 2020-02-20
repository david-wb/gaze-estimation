# Gaze Estimation with Deep Learning

This project implements a deep learning model to predict eye region landmarks and gaze direction.
The model is trained on a set of computer generated eye images synthesized with UnityEyes [1]. This work is heavily based on [2] with
some key modifications. 

### Setup

NOTE: This repo has been tested only on Ubuntu 16.04 and MacOS. 

First, create conda env for your system and activate it:
```bash
conda env create -f env-linux.yml
conda activate ge-linux
```

Then download the pretrained model files. One is for detecting face landmarks. The other is the main pytorch model.

```bash
./scripts/fetch_models.sh
```

Finally, run the webcam demo. You will likely need a GPU and have cuda 10.1 installed in order to get acceptable performance. 

```bash
python run_with_webcam.py
```

If you'd like to train the model yourself, please see the readme under `datasets/UnityEyes`.

### Materials and Methods

We generated over 100k training images using UnityEyes [1]. These images are each perfectly labeled
 with a json metadata file. The labels provide eye region landmark positions in screenspace,
  the direction the eye is looking in camera space, and other additional pieces of information. We extract from each
  training image, a region around normalized to the eye width. We then resize the
  extracted image to 150x90 pixels (WxH). For each preprocessed image, we create a set of heatmaps corresponding
  to 34 different eye region landmarks. The model is trained to predict the landmark locations and the direction of gaze
  in (pitch, yaw) form. The model is constructed and trained using pytorch. The overall method is summarized in the following figure.
![alt text](static/fig1.png "Logo Title Text 1")

The model architecture is based on the stacked hourglass model [3]. We added additional pre-hourglass layers to predict the gaze direction. 
The output of these additional layers is concatenated with the predicted eye-region landmarks before
being passed to two fully connected layers to predict the gaze direction. 

### Demo Video

[![Watch the video](static/ge_screenshot.png)](https://drive.google.com/open?id=1I0RLnd8QnFNU65Ov29B-tx_lc0GedSSB)


### References

1. https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/
2. https://github.com/swook/GazeML
3.  https://github.com/princeton-vl/pytorch_stacked_hourglass
