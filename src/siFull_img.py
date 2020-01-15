import tensorflow as tf
from siFinder import siFinder


def SI_full_img(x_dec, y_imgs, mask, patch_h, patch_w, ae_config, y_dec):

    N, C, H, W = x_dec.get_shape().as_list()

    # Transpose to channel last
    x_dec_t = tf.transpose(x_dec, (0, 2, 3, 1))
    y_imgs_t = tf.transpose(y_imgs, (0, 2, 3, 1))

    y_dec_t = tf.transpose(y_dec, (0, 2, 3, 1))

    for n in range(N):
        x_img = tf.expand_dims(x_dec_t[n], 0)
        y_img = tf.expand_dims(y_imgs_t[n], 0)
        y_img_dec = tf.expand_dims(y_dec_t[n], 0)

        # Extract patches
        x_patches = extract_patches(x_img, patch_h, patch_w)
        extracted_shape = x_patches.get_shape()
        x_patches = tf.reshape(x_patches, [-1, patch_h, patch_w, C])

        # Gives back a tensor of y_patches = si (channel last)
        y_patches, ncc, argmax_ncc, q, r, row, col = siFinder(x_patches, y_img, mask, 1, patch_h, patch_w, H, W,
                                                              ae_config, y_img_dec)

        # Reconstruct image
        y_patches = tf.reshape(y_patches, extracted_shape)

        # Notice that x_img is only passed to infer the right shape
        y_reconstructed = extract_patches_inverse(x_img, y_patches, patch_h, patch_w)

        # re-create the y batch
        if n == 0:
            y_syn = y_reconstructed
        else:
            y_syn = tf.concat([y_syn, y_reconstructed], axis=0)

    # Transpose back to channel first
    return tf.transpose(y_syn, (0, 3, 1, 2)), ncc, argmax_ncc, q, r, row, col, x_patches, y_patches


def extract_patches(x, patch_h, patch_w):
    ksize_rows = patch_h
    ksize_cols = patch_w

    # strides_rows and strides_cols determine the distance between the centers of two consecutive patches.
    strides_rows = patch_h  # no overlap
    strides_cols = patch_w

    # The size of sliding window
    ksizes = [1, ksize_rows, ksize_cols, 1]

    # How far the centers of 2 consecutive patches are in the image
    strides = [1, strides_rows, strides_cols, 1]

    return tf.extract_image_patches(x, ksizes, strides, rates=[1, 1, 1, 1], padding='SAME')


def extract_patches_inverse(x, y, patch_h, patch_w):
    _x = tf.zeros_like(x)
    _y = extract_patches(_x, patch_h, patch_w)
    grad = tf.gradients(_y, _x)[0]
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up
    return tf.gradients(_y, _x, grad_ys=y)[0] / grad
