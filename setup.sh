#! /bin/bash

unzip data/global-wheat-detection.zip -d data/
mv data/train.csv data/all.csv

mv utils/split_dataset.py split_dataset.py
python split_dataset.py
mv split_dataset.py utils/split_dataset.py