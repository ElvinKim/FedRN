# IID
# Noise type 'symmetric'
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.1  --save_dir cifar10/JointOpt/symmetric --iid --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.2  --save_dir cifar10/JointOpt/symmetric --iid --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.3  --save_dir cifar10/JointOpt/symmetric --iid --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.4  --save_dir cifar10/JointOpt/symmetric --iid --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.5  --save_dir cifar10/JointOpt/symmetric --iid --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.6  --save_dir cifar10/JointOpt/symmetric --iid --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20

python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.1  --save_dir cifar10/JointOpt/symmetric --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.2  --save_dir cifar10/JointOpt/symmetric  --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.3  --save_dir cifar10/JointOpt/symmetric --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.4  --save_dir cifar10/JointOpt/symmetric --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.5  --save_dir cifar10/JointOpt/symmetric --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type symmetric --noise_rate 0.6  --save_dir cifar10/JointOpt/symmetric --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20

# Noise type 'pairflip' --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.15  --save_dir cifar10/JointOpt/pairflip --iid --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.25  --save_dir cifar10/JointOpt/pairflip --iid --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.35  --save_dir cifar10/JointOpt/pairflip --iid --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.45  --save_dir cifar10/JointOpt/pairflip --iid --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20

# Noise type 'pairflip' --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.15  --save_dir cifar10/JointOpt/pairflip  --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.25  --save_dir cifar10/JointOpt/pairflip  --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.35  --save_dir cifar10/JointOpt/pairflip  --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20
python3 main_fed_LNL_JointOpt.py --dataset cifar --model mobile --epochs 300 --schedule 200 250 --save_dir cifar10 --noise_type pairflip --noise_rate 0.45  --save_dir cifar10/JointOpt/pairflip  --lr 0.1 --alpha 1.2 --beta 0.8 --begin 20