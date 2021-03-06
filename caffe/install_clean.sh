#! /bin/bash

DORN_ROOT=/home/markn1/DORN

source deactivate
conda remove -n caffe --all --yes
conda create -n caffe python=2.7 --yes
source activate caffe
cd $DORN_ROOT/caffe/python

for req in $(cat requirements.txt); do pip install $req; done

cd $DORN_ROOT/caffe
make clean
make all
make pycaffe
export PYTHONPATH=$DORN_ROOT/caffe/python:$DORN_ROOT/caffe/pylayer:$PYTHONPATH
pip install opencv-python
