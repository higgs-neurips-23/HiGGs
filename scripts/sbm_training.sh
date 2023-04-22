#!/bin/sh

conda init higgs
conda activate higgs

python dgd/main.py dataset=sbm dataset.h=1
python dgd/main.py dataset=sbm dataset.h=1.5
python dgd/main.py dataset=sbm dataset.h=2

