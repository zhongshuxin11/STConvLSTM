import os
import sys
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch
import logging
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from models.model_factory import Model
from utils import preprocess, logger
import trainer_loader as trainer
import datetime
from utils.early_stopping import EarlyStopping
from data_provider.moving_mnist import MovingMNIST
from utils.helper import adjust_learning_rate, AverageMeter

# -----------------------------------------------------------------------------
# torch.set_num_threads()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch STSF prediction model - STConvLSTM')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# data
# train_path = "/media/data/..."
# valid_path = "/media/data/..."

parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--train_data_paths', type=str, default=train_path)
parser.add_argument('--valid_data_paths', type=str, default=valid_path)
parser.add_argument('--save_dir', type=str, default='mnist/checkpoints/mnist_stconvlstm')
parser.add_argument('--gen_frm_dir', type=str, default='mnist/results/mnist_stconvlstm')
parser.add_argument('--loss_dir', type=str, default='mnist/loss/mnist_stconvlstm')
parser.add_argument('--print_path', type=str, default='mnist/loss/mnist_stconvlstm')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='stconvlstm')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
# add SpatiolBlock's parse
parser.add_argument('--sp_fusion', type=str, default='channel_add')
parser.add_argument('--sp_tln', type=int, default=0)
# loss function
parser.add_argument('--criterion', type=str, default='MSE')

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=30)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--adjust_interval', type=int, default=8)
parser.add_argument('--adjust_rate', type=float, default=0.5)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=2)
# shuxin have to revised, change point
parser.add_argument('--max_epoch', type=int, default=60)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=2)
parser.add_argument('--snapshot_interval', type=int, default=2)
parser.add_argument('--loss_interval', type=int, default=10)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()
print(args)


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def train_wrapper(model):
    # logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.print_path, 'train.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # early_stopping strategy
    early_stopping = EarlyStopping(patience=5, verbose=False)

    # record all the print content to txt file
    logger.make_print_to_file(args.print_path)
    if args.pretrained_model:
        model.load(args.pretrained_model)
    # load data
    train_inputs = MovingMNIST(args.train_data_paths,
                               sample_shape=(args.total_length, 1, args.img_width, args.img_width),
                               input_len=args.input_length)
    test_inputs = MovingMNIST(args.valid_data_paths,
                              sample_shape=(args.total_length, 1, args.img_width, args.img_width),
                              input_len=args.input_length)
    train_loaders = torch.utils.data.DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True,
                                                num_workers=8, pin_memory=True)
    test_loaders = torch.utils.data.DataLoader(test_inputs, batch_size=args.batch_size, shuffle=False,
                                               num_workers=8, pin_memory=True)

    # scheduled sampling setting
    eta = args.sampling_start_value

    # save traning loss and test loss
    train_loss = []
    test_loss = []
    test_ssim = []
    test_psnr = []
    test_fmae = []
    test_sharp = []
    test_iter = []

    llr = args.lr
    for epoch in tqdm(range(0, args.max_epoch + 1)):
        losses = AverageMeter()

        if epoch % args.adjust_interval == 0 and epoch > 0:
            llr = llr * args.adjust_rate
            model.optimizer = adjust_learning_rate(model.optimizer, llr)

        for ind, ims in enumerate(train_loaders):
            eta, real_input_flag = schedule_sampling(eta, epoch)

            ims = preprocess.reshape_patch_tensor(ims, args.patch_size)
            tr_loss = trainer.train(model, ims, real_input_flag, args, epoch)
            train_loss.append(tr_loss)

            losses.update(tr_loss)

            if ind % args.display_interval == 0:
                logging.info('[{0}][{1}]\t'
                             'lr: {lr:.5f}\t'
                             'loss: {loss.val:.6f} ({loss.avg:.6f})'.format(
                    epoch, ind, lr=model.optimizer.param_groups[-1]['lr'], loss=losses))

            torch.cuda.empty_cache()

            avg_mse, ssim, psnr, fmae, sharp = trainer.test(model, test_loaders, args, epoch)

        # plot figure to observe the losses
        x = range(len(train_loss))
        plt.figure(1)
        plt.title("this is losses of training")
        plt.plot(x, train_loss, label='loss')
        plt.legend()
        plt.savefig(args.loss_dir + '/train_loss.png')
        plt.close(1)
        # next

        if epoch % args.snapshot_interval == 0 and epoch > 0:
            model.save(epoch)

        if epoch % args.test_interval == 0 and epoch > 0:
            with torch.no_grad():
                avg_mse, ssim, psnr, fmae, sharp = trainer.test(model, test_loaders, args, epoch)
            test_iter.append(epoch)
            test_loss.append(avg_mse)
            test_ssim.append(ssim)
            test_psnr.append(psnr)
            test_fmae.append(fmae)
            test_sharp.append(sharp)
            early_stopping(avg_mse)

            # plot figure to observe the losses
            x = range(len(test_loss))
            plt.figure(1)
            plt.title("this is losses of validation")
            plt.plot(x, test_loss, label='loss')
            plt.legend()
            plt.savefig(args.loss_dir + '/valid_loss.png')
            plt.close(1)
            # next

        if epoch % args.loss_interval == 0 and epoch > 0:
            fileName = "/loss epoch{}".format(epoch) + datetime.datetime.now().strftime('date:' + '%Y_%m_%d')
            np.savez_compressed(args.loss_dir + fileName, train_loss=np.array(train_loss),
                                test_iter=np.array(test_iter), test_loss=np.array(test_loss),
                                test_ssim=np.array(test_ssim), test_psnr=np.array(test_psnr),
                                test_fmae=np.array(test_fmae), test_sharp=np.array(test_sharp))

        if early_stopping.early_stop:
            print("Early stopping")
            break

    fileName = "/loss all " + datetime.datetime.now().strftime('date:' + '%Y_%m_%d')
    np.savez_compressed(args.loss_dir + fileName, train_loss=np.array(train_loss), test_iter=np.array(test_iter),
                        test_loss=np.array(test_loss), test_ssim=np.array(test_ssim), test_psnr=np.array(test_psnr),
                        test_fmae=np.array(test_fmae), test_sharp=np.array(test_sharp))


def test_wrapper(model):
    # record all the print content to txt file
    logger.make_print_to_file(args.print_path)
    # test
    model.load(args.pretrained_model)
    # load data
    test_inputs = MovingMNIST(args.valid_data_paths,
                              sample_shape=(args.total_length, 1, args.img_width, args.img_width),
                              input_len=args.input_length)

    test_loaders = torch.utils.data.DataLoader(test_inputs, batch_size=args.batch_size, shuffle=False,
                                               num_workers=8, pin_memory=True)

    avg_mse, ssim, psnr, fmae, sharp = trainer.test(model, test_loaders, args, 'test_result')
    fileName = "/test loss " + datetime.datetime.now().strftime('date:' + '%Y_%m_%d')
    np.savez_compressed(args.loss_dir + fileName, avg_mse=avg_mse, ssim=ssim, psnr=psnr, fmae=fmae, sharp=sharp)


if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

if not os.path.exists(args.loss_dir):
    os.makedirs(args.loss_dir)

if not os.path.exists(args.print_path):
    os.makedirs(args.print_path)

print('Initializing models')

model = Model(args)

if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)
