# deceive-KG-models
An implementation of the experiments on KG robustness

## Getting Started

Clone the repository
```

git clone https://github.com/INK-USC/deceive-KG-models.git

```

## Question Answering

1. Download the pretrained models and datasets

```
cd deceive-KG-models/obqa

bash scripts/download.sh

cd data/cpnet

wget https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy
```

2. Preprocess the data

```
cd deceive-KG-models/obqa

python preprocess.py
```

By default, all available CPU cores will be used for multi-processing in order to speed up the process. Alternatively, you can use ```-p``` to specify the number of processes to use:

```
python preprocess.py -p 20
```

3. Train the base classifier

Then train a grn model (for example)using the command:

```
python grn.py -ds obqa --encoder bert-base-uncased -bs 64 -mbs 4 -dlr 1e-3
```
Similarly train RN and GN as well.

4. Train the triple classifier

```
python Get_neg_triples.py

python deep_triple_classifier.py
```

5. Pruning the Graph for only the useful nodes

```
python new_graph.py
```

6. Running Heuristics

```
python heuristics.py
```

The attributes are:

```-np --num_pert```:  number of perturbations

```--type``` type of perturbation(```rel``` for Relation Swapping, ```edge``` for Edge Deletion and ```edge1``` for Edge Rewiring)

For Relation Replacement use the command:

```
python train.py --mode_type eval --num_epochs 1 --save_dir ./saved_models/KG/model_25 --model_id 5 --enable_shuffle --dqn_lstm_len 100 --dqn_batch_size 16 --dqn_train_step 50 --log_path log_25.csv --steps_after_collecting_data 2000
```

7. Training the RL agent

```
python train.py --mode_type train --num_epochs 1 --save_dir ./saved_models/KG/model_25 --model_id 1 --enable_shuffle --dqn_lstm_len 100 --dqn_batch_size 16 --dqn_train_step 50 --log_path log_25.csv --steps_after_collecting_data 2000
```

8. Evaluating the RL agent

```
python train.py --mode_type eval --num_epochs 71801 --num_steps 70000 --save_dir ./saved_models/KG/model_25 --model_id 1 --debug_mode
```

what you have to change for specific model:

change of GPU number

change saved path to coincide with your saved model: ./saved_models/KG/model_25;  

change num_steps: the number of steps you want to perturb, 70000 as default

## Recommendation based experiments

```
cd deceive-KG-models/RippleNet
```

1. Train the base classifier

```
bash scripts/run_movie.sh
```
or 
```
bash scripts/run_music.sh
```

2. Run Heurisitics

```
python main_RN.py
```

The attributes are:
```--dataset```: ```movie``` or ```music```

```--type``` : Type of perturbation

```--num_pert``` : Number of perturbations

3. Train RL agent
```
python train.py --mode_type train --num_epochs 1 --save_dir ./saved_models/KG/model_25 --model_id 1 --enable_shuffle --dqn_lstm_len 100 --dqn_batch_size 16 --dqn_train_step 50 --log_path log_25.csv --steps_after_collecting_data 2000
```

4. Evaluate the RL agent
```
python train.py --mode_type eval --num_epochs 71801 --num_steps 70000 --save_dir ./saved_models/KG/model_25 --model_id 1 --debug_mode
```