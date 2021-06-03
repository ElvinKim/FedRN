python3 main_fed_LNL.py --method default --dataset cifar --model cnn4conv --epochs 500  --local_ep 10 --experiment case2 --noise_group_num 100 --group_noise_rate 0.4 --noise_type_lst pairflip  --save_dir cifar10/No_LNL/pairflip --lr 0.01 --partition label2  --forget_rate 0.4

# case 1
nsml run -e main_fed_LNL.py -d cifar-10 -m "symmetric non-iid 0.4 / selfie" -g 1 -c 1 --gpu-driver-version 440 --memory 7G -a "\
--dataset cifar
--noise_type_lst symmetric
--noise_group_num 100
--group_noise_rate 0.4
--forget_rate 0.4
--save_dir ./
--method selfie"

# case 2
nsml run -e main_fed_LNL.py -d cifar-10 -m "pairflip non-iid 0.4 / selfie" -g 1 -c 1 --gpu-driver-version 440 --memory 7G -a "\
--dataset cifar
--noise_type_lst pairflip
--noise_group_num 100
--group_noise_rate 0.4
--forget_rate 0.4
--save_dir ./
--method selfie"

# case 3
nsml run -e main_fed_LNL.py -d cifar-10 -m "symmetric non-iid 0.3-0.5 / selfie" -g 1 -c 1 --gpu-driver-version 440 --memory 7G -a "\
--dataset cifar
--noise_type_lst symmetric
--noise_group_num 50
--group_noise_rate 0.3 0.5
--forget_rate 0.5
--save_dir ./
--method selfie"

# case 4
nsml run -e main_fed_LNL.py -d cifar-10 -m "pairflip non-iid 0.3-0.5 / selfie" -g 1 -c 1 --gpu-driver-version 440 --memory 7G -a "\
--dataset cifar
--noise_type_lst pairflip
--noise_group_num 50
--group_noise_rate 0.3 0.5
--forget_rate 0.5
--save_dir ./
--method selfie"

# case 5
nsml run -e main_fed_LNL.py -d cifar-10 -m "mixed non-iid 0.3-0.5 / selfie" -g 1 -c 1 --gpu-driver-version 440 --memory 7G -a "\
--dataset cifar
--noise_type_lst symmetric pairflip
--noise_group_num 50 50
--group_noise_rate 0.3 0.5 0.3 0.5
--forget_rate 0.4
--save_dir ./
--method selfie"

