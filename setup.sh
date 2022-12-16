#!/usr/bin/bash

wget http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip trainDevTestTrees_PTB.zip

wget https://gist.githubusercontent.com/bastings/b094de2813da58056a05e8e7950d4ad1/raw/3fbd3976199c2b88de2ae62afc0ecc6f15e6f7ce/glove.840B.300d.sst.txt
wget https://gist.githubusercontent.com/bastings/4d1c346c68969b95f2c34cfbc00ba0a0/raw/76b4fefc9ef635a79d0d8002522543bc53ca2683/googlenews.word2vec.300d.txt
mkdir data
mv glove.840B.300d.sst.txt data/
mv googlenews.word2vec.300d.txt data/

mkdir models
