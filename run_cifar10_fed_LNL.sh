# IID
# Noise type 'symmetric'
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.1  --save_dir cifar10/No_LNL/symmetric --iid --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.2  --save_dir cifar10/No_LNL/symmetric --iid --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.3  --save_dir cifar10/No_LNL/symmetric --iid --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.4  --save_dir cifar10/No_LNL/symmetric --iid --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.5  --save_dir cifar10/No_LNL/symmetric --iid --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.6  --save_dir cifar10/No_LNL/symmetric --iid --lr 0.1

# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.1  --save_dir cifar10/No_LNL/symmetric --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.2  --save_dir cifar10/No_LNL/symmetric  --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.3  --save_dir cifar10/No_LNL/symmetric --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.4  --save_dir cifar10/No_LNL/symmetric --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.5  --save_dir cifar10/No_LNL/symmetric --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.6  --save_dir cifar10/No_LNL/symmetric --lr 0.1

# # Noise type 'pairflip'
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.15  --save_dir cifar10/No_LNL/pairflip --iid --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.25  --save_dir cifar10/No_LNL/pairflip --iid --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.35  --save_dir cifar10/No_LNL/pairflip --iid --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.45  --save_dir cifar10/No_LNL/pairflip --iid --lr 0.1

# # Noise type 'pairflip'
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.15  --save_dir cifar10/No_LNL/pairflip  --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.25  --save_dir cifar10/No_LNL/pairflip  --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.35  --save_dir cifar10/No_LNL/pairflip  --lr 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.45  --save_dir cifar10/No_LNL/pairflip  --lr 0.1


# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/No_LNL/IID_False/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.1
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/No_LNL/IID_False/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.2
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/No_LNL/IID_False/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.3
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/No_LNL/IID_False/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.4
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/No_LNL/IID_False/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.5


# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type pairflip --save_dir cifar10/No_LNL/IID_False/pairflip  --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.15
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type pairflip --save_dir cifar10/No_LNL/IID_False/pairflip  --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.25
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type pairflip --save_dir cifar10/No_LNL/IID_False/pairflip  --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.35
# python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type pairflip --save_dir cifar10/No_LNL/IID_False/pairflip  --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.45

python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/No_LNL/IID_False/symmetric --lr 0.1 --partition labeldir --noise_group_num 100 --group_noise_rate 0.1
python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/No_LNL/IID_False/symmetric --lr 0.1 --partition labeldir --noise_group_num 100 --group_noise_rate 0.2
python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/No_LNL/IID_False/symmetric --lr 0.1 --partition labeldir --noise_group_num 100 --group_noise_rate 0.3
python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/No_LNL/IID_False/symmetric --lr 0.1 --partition labeldir --noise_group_num 100 --group_noise_rate 0.4
python3 main_fed_LNL.py --dataset cifar --model mobile --epochs 500 --local_ep 2 --save_dir cifar10 --noise_type symmetric --save_dir cifar10/No_LNL/IID_False/symmetric --lr 0.1 --partition labeldir --noise_group_num 100 --group_noise_rate 0.5