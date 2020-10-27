import os.path
import datetime
import cv2
import torch
import numpy as np
from skimage.measure import compare_ssim
from utils import preprocess, metrics
from utils.hss import meteva_hss


def train(model, ims, real_input_flag, configs, itr=None):
    cost = model.train(ims, real_input_flag, itr)
    if configs.reverse_input:
        ims_rev = torch.flip(ims, [1])
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2
    return cost


def test(model, test_input_handle, configs, itr=None):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []
    hss = []

    output_length = configs.total_length - configs.input_length
    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)
        hss.append(0)

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    for ind, test_input in enumerate(test_input_handle):
        test_ims = test_input.numpy()    # test_ims shape: (batch, seq, channels, height, width)
        test_ims = np.transpose(test_ims, (0, 1, 3, 4, 2))
        batch_id = batch_id + 1

        test_dat = preprocess.reshape_patch_tensor(test_input, configs.patch_size)
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)  # img_gen: (8, 19, 64, 64, 1), numpy.ndarray
        img_out = img_gen[:, -output_length:]
        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            fmae[i] += metrics.batch_mae_frame_float(gx, x)
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse

            hss[i] += meteva_hss(x, gx, [0.28, 0.43, 0.58, 0.73])

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)
            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                ssim[i] += score
                sharp[i] += np.max(cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    fmae = np.asarray(fmae, dtype=np.float32) / batch_id
    sharp = np.asarray(sharp, dtype=np.float32) / (configs.batch_size * batch_id)
    hss = np.asarray(hss, dtype=np.float32) / batch_id

    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])
    print('fmae per frame: ' + str(np.mean(fmae)))
    for i in range(configs.total_length - configs.input_length):
        print(fmae[i])
    print('sharpness per frame: ' + str(np.mean(sharp)))
    for i in range(configs.total_length - configs.input_length):
        print(sharp[i])
    print('HSS per frame: ' + str(np.mean(hss)))
    for i in range(configs.total_length - configs.input_length):
        print(hss[i])

    return avg_mse, ssim, psnr, fmae, sharp
