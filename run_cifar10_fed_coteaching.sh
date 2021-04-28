# Noise type 'symmetric'
# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 100  --local_ep 10 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/coteaching/symmetric --iid --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.2 --forget_rate 0.2

# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/coteaching/symmetric --iid --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.2 --forget_rate 0.2

# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 1000  --local_ep 1 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/coteaching/symmetric --iid --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.2 --forget_rate 0.2

# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 100  --local_ep 10 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/coteaching/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.5 --forget_rate 0.5

# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/coteaching/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.5 --forget_rate 0.5

# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 1000  --local_ep 1 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/coteaching/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.5  --forget_rate 0.5


# # Noise type 'pairflip'
# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 100  --local_ep 10 --save_dir cifar10 --noise_type pairflip  --save_dir cifar10/coteaching/pairflip --iid --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.2  --forget_rate 0.2

# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type pairflip  --save_dir cifar10/coteaching/pairflip --iid --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.2 --forget_rate 0.2

# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 1000  --local_ep 1 --save_dir cifar10 --noise_type pairflip  --save_dir cifar10/coteaching/pairflip --iid --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.2 --forget_rate 0.2

# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 100  --local_ep 10 --save_dir cifar10 --noise_type pairflip  --save_dir cifar10/coteaching/pairflip --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.5 --forget_rate 0.5

# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type pairflip  --save_dir cifar10/coteaching/pairflip --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.5 --forget_rate 0.5

# python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 1000  --local_ep 1 --save_dir cifar10 --noise_type pairflip  --save_dir cifar10/coteaching/pairflip --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.5 --forget_rate 0.5

python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/coteaching/symmetric --iid --lr 0.1 --partition label2 --noise_group_num 50 50 --group_noise_rate 0.1 0.5 --forget_rate 0.1

python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/coteaching/symmetric --iid --lr 0.1 --partition label2 --noise_group_num 50 50 --group_noise_rate 0.1 0.5 --forget_rate 0.3

python3 main_fed_coteaching.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/coteaching/symmetric --iid --lr 0.1 --partition label2 --noise_group_num 50 50 --group_noise_rate 0.1 0.5 --forget_rate 0.5