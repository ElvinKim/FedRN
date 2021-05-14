# IID
# Noise type 'symmetric'
python3 main_fed_LNL.py --dataset mnist --model cnn --epochs 50 --schedule 30 40 --num_channels 1 --noise_type symmetric --noise_rate 0.2  --save_dir mnist --iid
python3 main_fed_LNL.py --dataset mnist --model cnn --epochs 50 --schedule 30 40 --num_channels 1 --noise_type symmetric --noise_rate 0.5  --save_dir mnist --iid

# Noise type 'pairflip'
python3 main_fed_LNL.py --dataset mnist --model cnn --epochs 50 --schedule 30 40 --num_channels 1 --noise_type pairflip --noise_rate 0.45  --save_dir mnist --iid


# # Non-IID
# # Noise type 'symmetric'
python3 main_fed_LNL.py --dataset mnist --model cnn --epochs 50 --schedule 30 40 --num_channels 1 --noise_type symmetric --noise_rate 0.2  --save_dir mnist
python3 main_fed_LNL.py --dataset mnist --model cnn --epochs 50 --schedule 30 40 --num_channels 1 --noise_type symmetric --noise_rate 0.5  --save_dir mnist

# # Noise type 'pairflip'
python3 main_fed_LNL.py --dataset mnist --model cnn --epochs 50 --schedule 30 40 --num_channels 1 --noise_type pairflip --noise_rate 0.45  --save_dir mnist