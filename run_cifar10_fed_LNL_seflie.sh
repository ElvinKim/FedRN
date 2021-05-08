nsml run -e main_fed_LNL.py -d cifar-10 -m "symmetric iid 0.1 / selfie" -g 1 -c 1 --memory 7G -a "\
--dataset cifar
--model mobile
--epochs 300
--schedule 200 250
--save_dir cifar10
--noise_type symmetric
--noise_group_num 100
--group_noise_rate 0.1
--save_dir ./
--iid
--lr 0.1
--method selfie"

nsml run -e main_fed_LNL.py -d cifar-10 -m "symmetric iid 0.2 / selfie" -g 1 -c 1 --memory 7G -a "\
--dataset cifar
--model mobile
--epochs 300
--schedule 200 250
--save_dir cifar10
--noise_type symmetric
--noise_group_num 100
--group_noise_rate 0.2
--save_dir ./
--iid
--lr 0.1
--method selfie"

nsml run -e main_fed_LNL.py -d cifar-10 -m "symmetric iid 0.3 / selfie" -g 1 -c 1 --memory 7G -a "\
--dataset cifar
--model mobile
--epochs 300
--schedule 200 250
--save_dir cifar10
--noise_type symmetric
--noise_group_num 100
--group_noise_rate 0.3
--save_dir ./
--iid
--lr 0.1
--method selfie"

nsml run -e main_fed_LNL.py -d cifar-10 -m "symmetric iid 0.4 / selfie" -g 1 -c 1 --memory 7G -a "\
--dataset cifar
--model mobile
--epochs 300
--schedule 200 250
--save_dir cifar10
--noise_type symmetric
--noise_group_num 100
--group_noise_rate 0.4
--save_dir ./
--iid
--lr 0.1
--method selfie"

nsml run -e main_fed_LNL.py -d cifar-10 -m "symmetric iid 0.5 / selfie" -g 1 -c 1 --memory 7G -a "\
--dataset cifar
--model mobile
--epochs 300
--schedule 200 250
--save_dir cifar10
--noise_type symmetric
--noise_group_num 100
--group_noise_rate 0.5
--save_dir ./
--iid
--lr 0.1
--method selfie"

nsml run -e main_fed_LNL.py -d cifar-10 -m "symmetric iid 0.6 / selfie" -g 1 -c 1 --memory 7G -a "\
--dataset cifar
--model mobile
--epochs 300
--schedule 200 250
--save_dir cifar10
--noise_type symmetric
--noise_group_num 100
--group_noise_rate 0.6
--save_dir ./
--iid
--lr 0.1
--method selfie"

nsml run -e main_fed_LNL.py -d cifar-10 -m "pairflip iid 0.15 / selfie" -g 1 -c 1 --memory 7G -a "\
--dataset cifar
--model mobile
--epochs 300
--schedule 200 250
--save_dir cifar10
--noise_type pairflip
--noise_group_num 100
--group_noise_rate 0.15
--save_dir ./
--iid
--lr 0.1
--method selfie"

nsml run -e main_fed_LNL.py -d cifar-10 -m "pairflip iid 0.25 / selfie" -g 1 -c 1 --memory 7G -a "\
--dataset cifar
--model mobile
--epochs 300
--schedule 200 250
--save_dir cifar10
--noise_type pairflip
--noise_group_num 100
--group_noise_rate 0.25
--save_dir ./
--iid
--lr 0.1
--method selfie"

nsml run -e main_fed_LNL.py -d cifar-10 -m "pairflip iid 0.35 / selfie" -g 1 -c 1 --memory 7G -a "\
--dataset cifar
--model mobile
--epochs 300
--schedule 200 250
--save_dir cifar10
--noise_type pairflip
--noise_group_num 100
--group_noise_rate 0.35
--save_dir ./
--iid
--lr 0.1
--method selfie"

nsml run -e main_fed_LNL.py -d cifar-10 -m "pairflip iid 0.45 / selfie" -g 1 -c 1 --memory 7G -a "\
--dataset cifar
--model mobile
--epochs 300
--schedule 200 250
--save_dir cifar10
--noise_type pairflip
--noise_group_num 100
--group_noise_rate 0.45
--save_dir ./
--iid
--lr 0.1
--method selfie"
