# Gaze Estimation with Deep Learning

This project implements a deep learning model to predict eye region landmarks and gaze direction.
The model is trained on a set of compute generated eye images synthesized using UnityEyes [1]. This work is heavily based on [2] with
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


### Demo Video

[![Watch the video](static/ge_screenshot.png)](https://drive.google.com/open?id=1I0RLnd8QnFNU65Ov29B-tx_lc0GedSSB)

### Methods

The overall method is summarized in the following figure.
![alt text](static/fig1.png "Logo Title Text 1")

