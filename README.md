# ScaleGUN

## setup
The default data path is "./data/".
```bash 
# build cython code
python setup.py build_ext --inplace
# generate removed edges/nodes
python dataProcessor.py --dataset cora
```

## Running the code
To run small graphs, use the following command:
```bash
python edge_exp.py --dataset cora --num_batch_removes 2000 --num_removes 1 --weight_mode test --disp 100 --lr 1 --rmax 1e-7 --dev 1 --edge_idx_start 0 --lam 1e-2 --std 0.1 --seed 0 &
```
For ogbn-papers100M, use the following command:
```bash
python edge_exp_large.py --dataset ogbn-papers100M --num_batch_removes 5 --num_removes 2000 --lam 1e-8 --weight_mode test --rmax 5e-9 --disp 1  --dev 1 --edge_idx_start 0 --lr 1 --std 5.0 --train_batch 32768 --epochs 400 --patience 30 --seed 0 &
``````
To run deep experiments, use the following command:
```bash
python deep_exp.py --dataset ogbn-arxiv --num_batch_removes 5 --num_removes 50 --lam 5e-4 --lr 1e-3 --weight_mode test --disp 1  --dev 1 --edge_idx_start 0 --patience 50 --layer 2 --train_batch 1024 --rmax 1e-7 --std 0.01 --seed 0
```
