# IID
# Noise type 'symmetric'
python3 main_fed_LG_finetuning.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/LG_finetuning/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.1 --forget_rate 0.1 --gpu 1

python3 main_fed_LG_finetuning.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/LG_finetuning/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.2 --forget_rate 0.2 --gpu 1

python3 main_fed_LG_finetuning.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/LG_finetuning/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.3 --forget_rate 0.3 --gpu 1

python3 main_fed_LG_finetuning.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/LG_finetuning/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.4 --forget_rate 0.4 --gpu 1

python3 main_fed_LG_finetuning.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/LG_finetuning/symmetric --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.5 --forget_rate 0.5 --gpu 1