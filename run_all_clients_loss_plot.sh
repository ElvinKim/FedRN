# coteaching
python3 main_fed_LNL.py --dataset cifar --epochs 101 --group_noise_rate 0.4 --method coteaching --forget_rate 0.4 --noise_type_lst symmetric --iid 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --group_noise_rate 0.4 --method coteaching --forget_rate 0.4 --noise_type_lst symmetric 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --group_noise_rate 0.4 --method coteaching --forget_rate 0.4 --noise_type_lst pairflip --iid 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --group_noise_rate 0.4 --method coteaching --forget_rate 0.4 --noise_type_lst pairflip

# selfie
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method selfie --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst symmetric --iid 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method selfie --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst symmetric 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method selfie --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst pairflip --iid 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method selfie --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst pairflip

# jointoptim
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method jointoptim --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst symmetric --iid 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method jointoptim --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst symmetric 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method jointoptim --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst pairflip --iid 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method jointoptim --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst pairflip

# dividemix
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method dividemix --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst symmetric --iid 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method dividemix --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst symmetric 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method dividemix --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst pairflip --iid 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method dividemix --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst pairflip

# default
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method default --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst symmetric --iid 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method default --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst symmetric 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method default --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst pairflip --iid 
python3 main_fed_LNL.py --dataset cifar --epochs 101 --method default --group_noise_rate 0.4 --forget_rate 0.4 --noise_type_lst pairflip