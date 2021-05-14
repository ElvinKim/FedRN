# IID
# Noise type 'symmetric'
python3 main_fedprox_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type clean --save_dir cifar10/FedProx/IID_False/clean --lr 0.1 --partition label2 
# python3 main_fedprox_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/FedProx/IID_False/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.2 --gpu 1
# python3 main_fedprox_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/FedProx/IID_False/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.3 --gpu 1
# python3 main_fedprox_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/FedProx/IID_False/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.4 --gpu 1
# python3 main_fedprox_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/FedProx/IID_False/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.5 --gpu 1


# python3 main_fedprox_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type pairflip --save_dir cifar10/FedProx/IID_False/pairflip  --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.15 --gpu 1
# python3 main_fedprox_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type pairflip --save_dir cifar10/FedProx/IID_False/pairflip  --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.25 --gpu 1
# python3 main_fedprox_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type pairflip --save_dir cifar10/FedProx/IID_False/pairflip  --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.35 --gpu 1
# python3 main_fedprox_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type pairflip --save_dir cifar10/FedProx/IID_False/pairflip  --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.45 --gpu 1