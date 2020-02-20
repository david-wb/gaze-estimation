# Gaze Estimation with Deep Learning

This project implements a deep learning model to predict eye region landmarks and gaze direction.
The model is trained on a set of computer generated eye images synthesized with UnityEyes [1]. This work is heavily based on [2] with
some key modifications. 

### Setup

First, create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate ge
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


### Demo Video

[![Watch the video](static/ge_screenshot.png)](https://drive.google.com/open?id=1I0RLnd8QnFNU65Ov29B-tx_lc0GedSSB)

### Methods

We generated over 100k training images using UnityEyes [1]. These images are each perfectly labeled
 with a json metadata file. The labels provide eye region landmarks points in screenspace and the look vector in camera space,
 and additionally other pieces of information such as head pose and lighting details.
The overall method is summarized in the following figure.
![alt text](static/fig1.png "Logo Title Text 1")

### References

1. https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/
2. https://github.com/swook/GazeML