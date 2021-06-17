import os
import csv
import nsml
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset

class Logger:
    def __init__(self, args, use_2_models):
        self.use_2_models = use_2_models

        # save results
        if args.save_dir is None:
            result_dir = './save/'
        else:
            result_dir = './save/{}/'.format(args.save_dir)

        if nsml.IS_ON_NSML:
            result_f = 'accuracy'
        else:
            result_f = 'fedLNL_{}_{}_{}_{}_C[{}]_BS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NGN[{}]_GNR[{}]_PT[{}]_EX[{}]_TPL[{}]_LAB[{}]'.format(
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
                args.experiment,
                args.T_pl,
                args.labeling
            )

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

        
            
def get_loss_dist(args, dataset, tmp_true_labels, net1, net2=None):
    if args.save_dir2 is None:
        result_dir = './save/'
    else:
        result_dir = './save/{}/'.format(args.save_dir2)

    result_f = 'loss_dist_model[{}]_method[{}]_noise{}_NR{}_IID[{}]_ep{}'.format(args.model, args.method, args.noise_type_lst, args.group_noise_rate, args.iid, args.g_epoch)            
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    f = open(result_dir + result_f + ".csv", 'w', newline='')
    wr = csv.writer(f)

    wr.writerow(['data_idx', 'is_noise', 'loss'])

    net1.eval() 
    if args.send_2_models:
        net2.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):   
            ce = nn.CrossEntropyLoss(reduce=False)
            images, labels = batch
            images, labels = images.to(args.device), labels.to(args.device)
            if args.send_2_models:
                logits1 = net1(images)
                logits2 = net2(images)
                loss1 = ce(logits1, labels)
                loss2 = ce(logits2, labels)
                loss = (loss1+loss2)/2
            else:
                if args.feature_return:
                    logits, feature = net1(images)
                else:
                    logits = net1(images)
                loss = ce(logits, labels)

            if tmp_true_labels[batch_idx] != labels:
                is_noise = 1
            else:
                is_noise = 0
            wr.writerow([batch_idx, is_noise, loss.item()])
    f.close()