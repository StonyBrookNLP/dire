#!/bin/bash

set -e
set -x

HOTPOTQA_DISTRACTOR_TRAIN=http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

wget $HOTPOTQA_DISTRACTOR_TRAIN
mv hotpot_train_v1.1.json data/raw/hotpot_train_v1.1.json

HOTPOTQA_DISTRACTOR_DEV=http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget $HOTPOTQA_DISTRACTOR_DEV
mv hotpot_dev_distractor_v1.json data/raw/hotpot_dev_distractor_v1.json
