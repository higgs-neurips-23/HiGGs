#!/bin/sh

conda init higgs
conda activate higgs

python dgd/main.py dataset=fb_hierarchies dataset.h=2  train.batch_size=8 dataset.resolution=10 train.n_epochs=500
python dgd/main.py dataset=fb_hierarchies dataset.h=1 dataset.resolution=10 train.batch_size=8 train.n_epochs=500
python dgd/main.py dataset=fb_hierarchies dataset.h=1.5 dataset.resolution=10 train.batch_size=3 train.n_epochs=250