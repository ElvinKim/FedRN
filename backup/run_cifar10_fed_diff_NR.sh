# IID
python3 main_fed_LNL_diff_NR.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10_diff_NR --iid --noise_group_num 50 50 --group_noise_rate 0.1 0.3 
python3 main_fed_LNL_diff_NR.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10_diff_NR --iid --noise_group_num 50 50 --group_noise_rate 0.1 0.5
        
        

# # Non-IID
python3 main_fed_LNL_diff_NR.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10_diff_NR --noise_group_num 50 50 --group_noise_rate 0.1 0.5
python3 main_fed_LNL_diff_NR.py --dataset cifar --model cnn --epochs 100 --schedule 50 75 --save_dir cifar10_diff_NR --noise_group_num 50 50 --group_noise_rate 0.1 0.3