# Distributed Training with PyG

This example is to show how you can use the PyG distributed framework to do distributed training over multiple nodes (partitions). We will show you the steps from generating the dataset partition to runn the distributed training by PyG distributed. 

## Requirements

- `python >= 3.6`
- `torch >= 1.12`

## Distributed (Multi-Node) Example

This example will show the distributed training on two datasets, 
1) Homo dataset, `ogbn-products`(http://snap.stanford.edu/ogb/data/nodeproppred/products.zip) from the [Open Graph Benchmark](https://ogb.stanford.edu/)
2) Hetero dataset,  'ogbn-mag'(http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip) from the [Open Graph Benchmark](https://ogb.stanford.edu/)

### Running the Example

#### Step 1: Prepare and partition the data

Here, enter the example/distributed/pyg folder and run the script below to get the homo partition and hetero partition

```bash
python partition_graph.py --dataset=ogbn-products --root_dir=./data/ogbn-products --num_partitions=2
```

```bash
python partition_hetero_graph.py --dataset=ogbn-mag --root_dir=./data/ogbn-mag --num_partitions=2
```


Also you can just run in default arguments with num_partitions=2 and datset.
```bash
python partition_graph.py
```
```bash
python partition_hetero_graph.py
```


#### Step 2: Run the example in each training node

For example, running the example in two nodes each with different process number by num_training_procs in one node:

```bash
# Node 0:
python dist_training_for_sage.py \
  --num_nodes=2 --node_rank=0 --num_training_procs=1 --master_addr=localhost \
  --dataset=ogbn-products --dataset_root_dir=./data/ogbn-products 

# Node 1:
python dist_training_for_sage.py \
  --num_nodes=2 --node_rank=1 --num_training_procs=1 --master_addr=localhost \
  --dataset=ogbn-products --dataset_root_dir=./data/ogbn-products
```



