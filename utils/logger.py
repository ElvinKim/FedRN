import os
import csv


class Logger:
    def __init__(self, args, use_2_models):
        self.use_2_models = use_2_models

        # save results
        if args.save_dir is None:
            result_dir = './save/'
        else:
            result_dir = './save/{}/'.format(args.save_dir)

        result_f = 'fedLNL_{}_{}_{}_{}_C[{}]_BS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NGN[{}]_GNR[{}]_PT[{}]'.format(
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
            args.noise_type,
            args.noise_group_num,
            args.group_noise_rate,
            args.partition,
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
