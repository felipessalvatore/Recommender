#!/bin/bash

DATA_DIR=$(pwd)/movielens
SIZE1=1m
SIZE10=10m
SIZE20=20m
mkdir -p ${DATA_DIR}
wget http://files.grouplens.org/datasets/movielens/ml-${SIZE1}.zip -O ${DATA_DIR}/ml-${SIZE1}.zip
unzip ${DATA_DIR}/ml-${SIZE1}.zip -d ${DATA_DIR}
wget http://files.grouplens.org/datasets/movielens/ml-${SIZE10}.zip -O ${DATA_DIR}/ml-${SIZE10}.zip
unzip ${DATA_DIR}/ml-${SIZE10}.zip -d ${DATA_DIR}
wget http://files.grouplens.org/datasets/movielens/ml-${SIZE20}.zip -O ${DATA_DIR}/ml-${SIZE20}.zip
unzip ${DATA_DIR}/ml-${SIZE20}.zip -d ${DATA_DIR}
