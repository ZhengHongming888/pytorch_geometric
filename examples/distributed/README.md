
Running example for 2 partitions: 

CUDA_VISIBLE_DEVICES=0 python dist_train_sage_in_work_mode.py --dataset_root_dir=/lfs/lfs14/xxx/products0 --num_nodes=2 --node_rank=0 --num_training_procs=1 --master_addr=IP addr

CUDA_VISIBLE_DEVICES=0 python dist_train_sage_in_work_mode.py --dataset_root_dir=/lfs/lfs14/xxx/products0 --num_nodes=2 --node_rank=1 --num_training_procs=1 --master_addr=IP addr