#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd ${DIR}/..

# Download face landmark predictor model
wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
bzip2 -d shape_predictor_5_face_landmarks.dat.bz2
rm shape_predictor_5_face_landmarks.dat.bz2


# Download trained pytorch model
wget "https://drive.google.com/uc?export=download&id=17aJAUAIl-1VPvJcPeahH8MQrcLRpy9Li" -O checkpoint.pt
