# IID
# Noise type 'symmetric'
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.1  --save_dir cifar10/gfilter/symmetric --iid
# python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.2  --save_dir cifar10/gfilter/symmetric --iid
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.3  --save_dir cifar10/gfilter/symmetric --iid
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.4  --save_dir cifar10/gfilter/symmetric --iid
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.5  --save_dir cifar10/gfilter/symmetric --iid
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.6  --save_dir cifar10/gfilter/symmetric --iid


# # # Non-IID
# # # Noise type 'symmetric'
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.1  --save_dir cifar10/gfilter/symmetric
# python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.2  --save_dir cifar10/gfilter/symmetric
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.3  --save_dir cifar10/gfilter/symmetric
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.4  --save_dir cifar10/gfilter/symmetric 
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.5  --save_dir cifar10/gfilter/symmetric
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --noise_type symmetric --noise_rate 0.6  --save_dir cifar10/gfilter/symmetric


# # # Noise type 'pairflip'
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type pairflip --noise_rate 0.15  --save_dir cifar10/gfilter/pairflip --iid
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type pairflip --noise_rate 0.25  --save_dir cifar10/gfilter/pairflip --iid
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type pairflip --noise_rate 0.35  --save_dir cifar10/gfilter/pairflip --iid
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type pairflip --noise_rate 0.45  --save_dir cifar10/gfilter/pairflip --iid

python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type pairflip --noise_rate 0.15  --save_dir cifar10/gfilter/pairflip
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type pairflip --noise_rate 0.25  --save_dir cifar10/gfilter/pairflip
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type pairflip --noise_rate 0.35  --save_dir cifar10/gfilter/pairflip
python3 main_fed_LNL_Gfiltering.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10 --noise_type pairflip --noise_rate 0.45  --save_dir cifar10/gfilter/pairflip



