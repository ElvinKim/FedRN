# IID
python3 main_fed.py --dataset mnist --model cnn --epochs 50 --schedule 30 40 --num_channels 1 --save_dir mnist --iid

# Non-IID
python3 main_fed.py --dataset mnist --model cnn --epochs 50 --schedule 30 40 --num_channels 1 --save_dir mnist