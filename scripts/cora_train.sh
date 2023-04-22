#!/bin/sh

conda init higgs
conda activate higgs

python dgd/main.py dataset=cora dataset.h=2   dataset.resolution=3 train.n_epochs=500
python dgd/main.py dataset=cora dataset.h=1 dataset.resolution=1 train.batch_size=4 train.n_epochs=1000
python dgd/main.py dataset=cora dataset.h=1.5 dataset.resolution=1 train.batch_size=2 train.n_epochs=500


