# IID
# Noise type 'symmetric'
python3 main_fed_LNL.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type symmetric --noise_rate 0.2  --save_dir cifar10 --iid
python3 main_fed_LNL.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type symmetric --noise_rate 0.5  --save_dir cifar10 --iid


# Noise type 'pairflip'
python3 main_fed_LNL.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type pairflip --noise_rate 0.45  --save_dir cifar10 --iid

# # Non-IID
# # Noise type 'symmetric'
python3 main_fed_LNL.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type symmetric --noise_rate 0.2  --save_dir cifar10
python3 main_fed_LNL.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type symmetric --noise_rate 0.5  --save_dir cifar10


# # Noise type 'pairflip'
python3 main_fed_LNL.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type pairflip --noise_rate 0.45  --save_dir cifar10

