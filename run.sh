# -- CIFAR 10 -- #

# Non-IID Shard 2 (Symmetric 0.0-0.4)

# FedRN
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--w_alpha 0.6 \
--partition shard \
--num_shards 200 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method dividemix


# Non-IID Shard 2 (Asymmetric 0.0-0.4)

# FedRN
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--w_alpha 0.6 \
--partition shard \
--num_shards 200 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method dividemix

# Non-IID Shard 2 (Mixed 0.0-0.4)

# FedRN
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--w_alpha 0.6 \
--partition shard \
--num_shards 200 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition shard \
--num_shards 200 \
--method dividemix


# Non-IID Dirichlet (0.5) (Symmetric 0.0-0.4)
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--w_alpha 0.6 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method dividemix


# Non-IID Dirichlet (0.5) (Asymmetric 0.0-0.4)

# FedRN
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--w_alpha 0.6 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method dividemix

# Non-IID Dirichlet (0.5) (Mixed 0.0-0.4)
# FedRN
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--w_alpha 0.6 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method dividemix


# -- CIFAR 100 -- #

# Non-IID Shard 20 (Symmetric 0.0-0.4)

# FedRN
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--w_alpha 0.6 \
--partition shard \
--num_shards 2000 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method dividemix


# Non-IID Shard 20 (Asymmetric 0.0-0.4)

# FedRN
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--w_alpha 0.6 \
--partition shard \
--num_shards 2000 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method dividemix

# Non-IID Shard 20 (Mixed 0.0-0.4)

# FedRN
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--w_alpha 0.6 \
--partition shard \
--num_shards 2000 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition shard \
--num_shards 2000 \
--method dividemix


# Non-IID Dirichlet (0.5) (Symmetric 0.0-0.4)
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--w_alpha 0.6 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method dividemix


# Non-IID Dirichlet (0.5) (Asymmetric 0.0-0.4)

# FedRN
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--w_alpha 0.6 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method dividemix

# Non-IID Dirichlet (0.5) (Mixed 0.0-0.4)
# FedRN
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--w_alpha 0.6 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method fedrn 

# Co-Teaching
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method coteaching

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method jointoptim

# Joint Optimization
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method SELFIE

# DivdeMix
python main_fed_LNL.py \
--dataset cifar100 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip \
--noise_group_num 50 50 \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partitioni dirichlet \
--dd_alpha 0.5 \
--method dividemix
