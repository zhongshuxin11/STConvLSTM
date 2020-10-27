__author__ = 'shuxin'

import os
import shutil
import argparse
import numpy as np
import torch
from data_provider import datasets_factory
from models.model_factory import Model
from utils import preprocess, logger
import radar_trainer as trainer
import datetime
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
torch.set_num_threads(8)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device_ids = [0, 2]
parser = argparse.ArgumentParser(description='PyTorch video prediction model - STConvLSTM')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# data
# train_path = "/media/data/data_entry/zhongsx/code/data/radar/radar_train_40Fnew_rm.npz"
# valid_path = "/media/data/data_entry/zhongsx/code/data/radar/radar_valid_40Fnew_rm.npz"
# data
train_path = "/media/data/data_entry/zhongsx/code/data/radar/radar_train_30seq.npz"
valid_path = "/media/data/data_entry/zhongsx/code/data/radar/radar_valid_30seq.npz"

parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--train_data_paths', type=str, default=train_path)
parser.add_argument('--valid_data_paths', type=str, default=valid_path)
parser.add_argument('--save_dir', type=str, default='radar/checkpoints/radar30seq_stconvlstm_L64_iter')
parser.add_argument('--gen_frm_dir', type=str, default='radar/results/radar30seq_stconvlstm_L64_iter')
parser.add_argument('--loss_dir', type=str, default='radar/loss/radar30seq_stconvlstm_L64_iter')
# the print content save path, ect training loss
parser.add_argument('--print_path', type=str, default='radar/loss/radar30seq_stconvlstm_L64_iter')
# input & output size
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=140)
parser.add_argument('--img_channel', type=int, default=1)
# model
parser.add_argument('--model_name', type=str, default='stconvlstm')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')  # 64, 64, 64, 64
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)  # 4
parser.add_argument('--layer_norm', type=int, default=1)
# add SpatiolBlock's parse
parser.add_argument('--sp_fusion', type=str, default='channel_add')
parser.add_argument('--sp_tln', type=int, default=0)        # tln = =0, SNL; tln =1, GC transform

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)  # 5000
parser.add_argument('--sampling_start_value', type=float, default=1.0)  # 1.0
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.01)  ## try 0.001, 0.01, 0.1
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)  # 8
parser.add_argument('--max_iterations', type=int, default=80000)  # 80000, 20000
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5000)  # 5000
parser.add_argument('--snapshot_interval', type=int, default=5000)  # 5000, checkpoint save
parser.add_argument('--loss_interval', type=int, default=5000)
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
    # record all the print content to txt file
    logger.make_print_to_file(args.print_path)
    if args.pretrained_model:
        model.load(args.pretrained_model)
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=True)

    # schedule sampling
    eta = args.sampling_start_value
    # not use schedule sampling, make real_input_flag
    """
    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    """

    ### save traning loss and test loss
    train_loss = []
    test_loss = []
    test_ssim = []
    test_psnr = []
    test_fmae = []
    test_sharp = []
    test_iter = []

    for itr in range(1, args.max_iterations + 1):
        # reshape into: (batch_size, seq_length, height/patch, width/patch, patch_size*patch_size*num_channels)

        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        ims = preprocess.reshape_patch(ims, args.patch_size)

        eta, real_input_flag = schedule_sampling(eta, itr)

        tr_loss = trainer.train(model, ims, real_input_flag, args, itr)
        train_loss.append(tr_loss)

        if itr % args.snapshot_interval == 0:
            model.save(itr)

        if itr % args.test_interval == 0:
            avg_mse, ssim, psnr, fmae, sharp = trainer.test(model, test_input_handle, args, itr)
            test_iter.append(itr)
            test_loss.append(avg_mse)
            test_ssim.append(ssim)
            test_psnr.append(psnr)
            test_fmae.append(fmae)
            test_sharp.append(sharp)

            x = range(len(train_loss))
            plt.figure(1)
            plt.title("this is losses of training")
            plt.plot(x, train_loss, label='loss')
            plt.legend()
            plt.savefig(args.loss_dir + '/train_loss.png')
            plt.close(1)

            # plot figure to observe the losses
            x = range(len(test_loss))
            plt.figure(1)
            plt.title("this is losses of validation")
            plt.plot(x, test_loss, label='loss')
            plt.legend()
            plt.savefig(args.loss_dir + '/valid_loss.png')
            plt.close(1)
            # next

        if itr % args.loss_interval == 0:
            fileName = "/loss iter{}".format(itr) + datetime.datetime.now().strftime('date:' + '%Y_%m_%d')
            np.savez_compressed(args.loss_dir + fileName, train_loss=np.array(train_loss),
                                test_iter=np.array(test_iter), test_loss=np.array(test_loss),
                                test_ssim=np.array(test_ssim), test_psnr=np.array(test_psnr),
                                test_fmae=np.array(test_fmae), test_sharp=np.array(test_sharp))

        train_input_handle.next()

    fileName = "/loss all " + datetime.datetime.now().strftime('date:' + '%Y_%m_%d')
    np.savez_compressed(args.loss_dir + fileName, train_loss=np.array(train_loss), test_iter=np.array(test_iter),
                        test_loss=np.array(test_loss), test_ssim=np.array(test_ssim), test_psnr=np.array(test_psnr),
                        test_fmae=np.array(test_fmae), test_sharp=np.array(test_sharp))




def test_wrapper(model):
    # record all the print content to txt file
    logger.make_print_to_file(args.print_path)
    # test
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=False)
    avg_mse, ssim, psnr, fmae, sharp = trainer.test(model, test_input_handle, args, 'test_result')
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

# gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
# args.n_gpu = len(gpu_list)
print('Initializing models')

model = Model(args)
"""
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(args.device)
"""
if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)
