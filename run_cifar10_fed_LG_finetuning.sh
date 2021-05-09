# IID
# Noise type 'symmetric'
python3 main_fed_LG_finetuning.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/LG_finetuning/symmetric --iid --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.2 --forget_rate 0.2

python3 main_fed_LG_finetuning.py --dataset cifar --model mobile --epochs 500  --local_ep 2 --save_dir cifar10 --noise_type symmetric  --save_dir cifar10/LG_finetuning/symmetric --iid --lr 0.1 --partition label2 --noise_group_num 100 --group_noise_rate 0.5 --forget_rate 0.5