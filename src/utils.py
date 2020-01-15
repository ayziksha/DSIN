import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import skimage.measure
import ms_ssim_np_imgcomp
import scipy.stats
from skimage.util import view_as_windows
from PIL import Image
import os


def plot_loss(train_loss_history, val_loss_history, val_iters, train_iters, total_iterations, best_val, iter, model_name):
    """ plot train+ val loss """

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 16
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size

    plt.plot(train_iters, train_loss_history, '.')
    plt.plot(val_iters, val_loss_history, '.')
    plt.xlim([0, total_iterations])

    plt.title('Train and Validation - average loss per iteration')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.suptitle('Best validation loss = ' + str(best_val) +
                 ', Best validation iterations = ' + str(iter) + '/' + str(total_iterations) +
                 '\nModel name = ' + str(model_name))
    plt.subplots_adjust(hspace=1)
    plt.show(block=True)


def plot_inference(x, x_dec, y, y_syn, x_with_si, model_name, iter, cnt='NA', lr='NA', bpp='NA'):
    """ Plot orig x, orig y, synthetic y and diff """

    # order to channels last
    x = np.transpose(x, (1, 2, 0))
    x_dec = np.transpose(x_dec, (1, 2, 0))
    y = np.transpose(y, (1, 2, 0))
    y_syn = np.transpose(y_syn, (1, 2, 0))
    x_with_si = np.transpose(x_with_si, (1, 2, 0))

    diff_no_si, spatial_loss_no_si = l1_x_vs_rec(x, x_dec)
    diff_with_si, spatial_loss_si = l1_x_vs_rec(x, x_with_si)
    psnr_no_si = psnr_x_vs_rec(x, x_dec)
    psnr_si = psnr_x_vs_rec(x, x_with_si)
    msssim_no_si = msssim_x_vs_rec(x, x_dec)
    msssim_si = msssim_x_vs_rec(x, x_with_si)

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 18
    fig_size[1] = 11
    plt.rcParams["figure.figsize"] = fig_size

    plt.subplot(321)
    plt.imshow(x)
    plt.title('original x')
    plt.subplot(323)
    plt.imshow(y_syn.astype('uint8'))
    plt.title('synthetic y')
    plt.subplot(325)
    plt.imshow(y)
    plt.title('original y')
    plt.subplot(222)
    plt.imshow(x_dec.astype('uint8'))
    plt.title('x decoded')
    plt.subplot(224)
    plt.imshow(x_with_si.astype('uint8'))
    plt.title('x_with_si')

    plt.suptitle('x_no_si: l1 = ' + str(spatial_loss_no_si) + ', psnr = ' + str(psnr_no_si) + ', ms-ssim = ' + str(
        msssim_no_si) + '\nx_with_si: l1 = ' + str(spatial_loss_si) + ', psnr = ' + str(psnr_si) + ', ms-ssim = ' + str(
        msssim_si) + '\nae_lr = ' + str(lr[0]) + ', pc_lr = ' + str(lr[1]) + ', iters = ' + str(cnt) + '/' + str(
        iter) + ', bpp = ' + str(bpp) + '\nModel Name = ' + str(model_name))

    plt.subplots_adjust(top=0.8)
    plt.show(block=True)


def l1_x_vs_rec(x, x_rec):
    """ L1 loss between orig x and x_rec """
    diff = x.astype('float32') - x_rec.astype('float32')
    diff = np.absolute(diff)
    spatial_l1_loss = np.mean(diff)
    return diff.astype('uint8'), spatial_l1_loss


def psnr_x_vs_rec(x, x_rec):
    return np.float32(skimage.measure.compare_psnr(x, x_rec.astype('uint8')))


def msssim_x_vs_rec(x, x_rec):
    if x.ndim < 4:
        x = np.expand_dims(x, axis=-1)
    if x_rec.ndim < 4:
        x_rec = np.expand_dims(x_rec, axis=-1)
    return ms_ssim_np_imgcomp._calc_msssim_orig(x, x_rec)


def save_test_imgs_fn(root_save_img, model_name, x_with_si, i, bpp):
    if not os.path.exists(root_save_img):
        os.mkdir(root_save_img)

    img_save_path = root_save_img + model_name
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    img_final = Image.fromarray(np.transpose(x_with_si, (1, 2, 0)).astype('uint8'), 'RGB')
    img_final.save(img_save_path + '/' + str(i) + '_' + '{:.5f}bpp.png'.format(bpp))


def loss_list_saver(x, y, x_rec, y_syn, batch_size, model_name, bpp, root_save_img):
    """Saves a list of L1 losses in txt file"""

    # order to channels last
    x = np.transpose(x, (0, 2, 3, 1))
    y = np.transpose(y, (0, 2, 3, 1))
    x_rec = np.transpose(x_rec, (0, 2, 3, 1))
    y_syn = np.transpose(y_syn, (0, 2, 3, 1))

    # x vs x_rec
    f1 = open(root_save_img + 'bpp_list_' + model_name + '.txt', 'a+')
    f2 = open(root_save_img + 'l1_list_' + model_name + '.txt', 'a+')
    f3 = open(root_save_img + 'psnr_list_' + str(model_name) + '.txt', 'a+')
    f4 = open(root_save_img + 'msssim_list_' + str(model_name) + '.txt', 'a+')

    f5 = open(root_save_img + 'mse_list_x_y_syn_' + str(model_name) + '.txt', 'a+')
    f6 = open(root_save_img + 'avg_Pearson_list_x_y_syn_' + str(model_name) + '.txt', 'a+')

    for i in range(batch_size):
        # x vs x_rec
        f1.write(str(bpp) + '\n')  # bpp

        _, loss = l1_x_vs_rec(x[i], x_rec[i])  # L1 x and x_rec
        f2.write(str(loss) + '\n')

        psnr = psnr_x_vs_rec(x[i], x_rec[i])  # psnr x and x_rec
        f3.write(str(psnr) + '\n')

        msssim = msssim_x_vs_rec(x[i], x_rec[i])  # mssim x and x_rec
        f4.write(str(msssim) + '\n')

        mse = np.mean((x[i].astype('float32') - y_syn[i].astype('float32')) ** 2)
        f5.write(str(mse) + '\n')

        avg_Pearson = pearson_per_patch(x[i], y_syn[i])
        f6.write(str(avg_Pearson) + '\n')

    # x vs x_rec
    f1.close()
    f2.close()
    f3.close()
    f4.close()

    f5.close()
    f6.close()


def pearson_per_patch(x, y):
    """ Calc average Pearson for each patch in x and its matching patch in y_syn """
    patch_h, patch_w = 20, 24
    patches_x = view_as_windows(x, [patch_h, patch_w, 3],
                                [patch_h, patch_w, 3])  # inputs: (im, [window shape], [stride in each axis])
    patches_y = view_as_windows(y, [patch_h, patch_w, 3], [patch_h, patch_w, 3])

    a, b, _, _, _, _ = np.shape(patches_x)
    num_patches = a * b

    patches_x_reshaped = patches_x.reshape(num_patches, -1)
    patches_y_reshaped = patches_y.reshape(num_patches, -1)

    tot = 0.0
    for i in range(num_patches):
        patch_pearson, _ = scipy.stats.pearsonr(patches_x_reshaped[i], patches_y_reshaped[i])
        tot += patch_pearson
    avg_pearson = tot/num_patches

    return avg_pearson
