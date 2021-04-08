# deceive-KG-models
An implementation of the experiments on KG robustness

git clone https://github.com/INK-USC/deceive-KG-models.git

cd deceive-KG-models/obqa

bash scripts/download.sh


cd data/cpnet

wget https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy

cd ~

cd deceive-KG-models/obqa


python preprocess.py

By default, all available CPU cores will be used for multi-processing in order to speed up the process. Alternatively, you can use "-p" to specify the number of processes to use:

python preprocess.py -p 20


Then train a grn model using the command:
python grn.py -ds obqa --encoder bert-base-uncased -bs 64 -mbs 4 -dlr 1e-3

TRAINING THE TRIPLE CLASSIFIER

python Get_neg_triples.py

python deep_triple_classifier.py


RUNNING HEURISTICS

