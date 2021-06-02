import os
import csv
import numpy as np
import nsml


class Logger:
    def __init__(self, args, use_2_models):
        self.use_2_models = use_2_models

        # save results
        if args.save_dir is None:
            result_dir = './save/'
        else:
            result_dir = './save/{}/'.format(args.save_dir)

<<<<<<< HEAD
        result_f = '{}_{}_{}_{}_C[{}]_BS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NGN[{}]_GNR[{}]_PT[{}]_EX[{}]'.format(
            args.dataset,
            args.method,
            args.model,
            args.epochs,
            args.frac,
            args.local_bs,
            args.local_ep,
            args.iid,
            args.lr,
            args.momentum,
            args.noise_type_lst,
            args.noise_group_num,
            args.group_noise_rate,
            args.partition,
            args.experiment
        )
        
        if args.method in ['coteaching', 'coteaching+', 'finetune', 'lgfinetune', 'gfilter', 'gmix', 'lgteaching']:
            result_f += "_FR[{}]_FRS[{}]".format(args.forget_rate, args.forget_rate_schedule)
=======
        if nsml.IS_ON_NSML:
            result_f = 'accuracy'
        else:
            result_f = 'fedLNL_{}_{}_{}_{}_C[{}]_BS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NGN[{}]_GNR[{}]_PT[{}]_EX[{}]'.format(
                args.dataset,
                args.method,
                args.model,
                args.epochs,
                args.frac,
                args.local_bs,
                args.local_ep,
                args.iid,
                args.lr,
                args.momentum,
                args.noise_type_lst,
                args.noise_group_num,
                args.group_noise_rate,
                args.partition,
                args.experiment
            )
>>>>>>> a1f51a5d22ef800621910758bf0b9a15f11ff2db

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.f = open(result_dir + result_f + ".csv", 'w', newline='')
        self.wr = csv.writer(self.f)

        if self.use_2_models:
            self.wr.writerow(['epoch', 'train_acc', 'train_loss', 'train_acc2', 'train_loss2',
                              'test_acc', 'test_loss', 'test_acc2', 'test_loss2'])
        else:
            self.wr.writerow(['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])

        # Option Save
        with open(result_dir + result_f + ".txt", 'w') as option_f:
            option_f.write(str(args))

    def write(self, epoch, train_acc, train_loss, test_acc, test_loss,
              train_acc2=None, train_loss2=None, test_acc2=None, test_loss2=None):

        if self.use_2_models:
            self.wr.writerow([epoch, train_acc, train_loss, train_acc2, train_loss2,
                              test_acc, test_loss, test_acc2, test_loss2])
        else:
            self.wr.writerow([epoch, train_acc, train_loss, test_acc, test_loss])

    def close(self):
        self.f.close()


class NoiseLogger:
    def __init__(self, args, user_noisy_data, dict_users):
        # save results
        if args.save_dir is None:
            result_dir = './save/'
        else:
            result_dir = './save/{}/'.format(args.save_dir)

        if nsml.IS_ON_NSML:
            result_f = 'noise_label_accuracy'
        else:
            result_f = 'fedLNL_noise_{}_{}_{}_{}_C[{}]_BS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NGN[{}]_GNR[{}]_PT[{}]_EX[{}]'.format(
                args.dataset,
                args.method,
                args.model,
                args.epochs,
                args.frac,
                args.local_bs,
                args.local_ep,
                args.iid,
                args.lr,
                args.momentum,
                args.noise_type_lst,
                args.noise_group_num,
                args.group_noise_rate,
                args.partition,
                args.experiment
            )

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.f = open(result_dir + result_f + ".csv", 'w', newline='')
        self.wr = csv.writer(self.f)

        self.wr.writerow(['epoch', 'user_id', 'total_epochs', 'precision', 'recall'])

        self.user_total_data = dict_users
        self.user_noisy_data = user_noisy_data
        self.user_clean_data = {user: list(set(total_data) - set(user_noisy_data[user]))
                                for user, total_data in dict_users.items()}

        self.clean_ids = []

    def update(self, ids):
        self.clean_ids.extend(ids)

    def write(self, epoch, user_id, total_epochs):
        # precision is computed through
        # # of truly detected noisy labels / total # of detected noisy labels

        if self.clean_ids:
            self.clean_ids = np.array(self.clean_ids)
            pred_noisy_data = set(self.user_total_data[user_id]) - set(self.clean_ids)
        else:
            pred_noisy_data = []

        noisy_data = self.user_noisy_data[user_id]
        correct_pred_noisy_data = set(pred_noisy_data) & set(noisy_data)

        precision = len(correct_pred_noisy_data) / max(len(pred_noisy_data), 1)
        recall = len(correct_pred_noisy_data) / max(len(noisy_data), 1)

        self.wr.writerow([epoch, user_id, total_epochs, precision, recall])

        self.clean_ids = []

    def close(self):
        self.f.close()
