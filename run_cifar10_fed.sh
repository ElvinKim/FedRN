# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 100  --local_ep 10 --save_dir cifar10 --noise_type clean  --save_dir cifar10/clean --iid --lr 0.1 

# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type clean  --save_dir cifar10/clean --iid --lr 0.1

# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 1000 --local_ep 1 --save_dir cifar10 --noise_type clean  --save_dir cifar10/clean --iid --lr 0.1


# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 100 --local_ep 10 --save_dir cifar10 --noise_type clean  --save_dir cifar10/clean  --lr 0.1 

# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type clean  --save_dir cifar10/clean  --lr 0.1

# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 1000 --local_ep 1 --save_dir cifar10 --noise_type clean  --save_dir cifar10/clean  --lr 0.1

python3 main_fed.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type clean  --save_dir cifar10/clean --lr 0.1 --partition label2 --seed 1000