# Base-$(k+1)$ Graph
We propose the Base-(k+1) Graph, which is finite-time convergence for any number of nodes and maximum degree $k$,
Thanks this property, the Base-(k+1) Graph enables Decentralized SGD to converge faster with fewer communication costs than the exponential graph.

## Dependency
``
conda env create --file conda_env.yaml
conda activate MT_env
``

## Visualization
```
from base_graph import *
graph = BaseGraph(5, 1) # 5 nodes and maximum degree 1.
graph.visualize()
```

Moreover, the mixing matrices of the Base-(k+1) Graph can be get as follows.
```
print(graph.w_list)
```

## Decentralized Learning on Base-(k+1) Graph
We provide the implementation of the Base-(k+1) Graph with Decentralized SGD.
You can run the Base-2 Graph, whose maximum degree is 1, with Decentralized SGD by the following command.
```
python evaluate.py ${log_path} --model vgg --optimizer gossip --dataset cifar10 --seed 0 --config ./config/25_node.json  --node_list 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --nw one_peer_base --lr 0.1 --epoch 500 --alpha 0.1 --beta 0.9 --local_step 1 --batch 32
```
The configuration file `./config/25_node.json` is set up for an environment with eight GPUs.
Please rewrite it to your environment accordingly.

To reproduce the experimental results in our paper, we provide the bash script.
```
bash evalute_cifar10.sh
```