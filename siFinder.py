import tensorflow as tf

"""This function takes x_dec and full image of y and outputs y_syn by looking for highest correlation between them
This is a non trainable function """


def siFinder(x_patches_orig, y_images_orig, mask, batch_size, patch_size_h, patch_size_w, H, W, ae_config, y_dec):

    x_patches = x_patches_orig
    y_images = y_images_orig

    # Normalize images for normalized cross correlation:
    q = rgb_transform(x_patches, ae_config) if ae_config.use_L2andLAB \
        else rgb_transform(reduce_mean_and_std_normalize_images(x_patches, ae_config), ae_config)

    r = rgb_transform(y_dec, ae_config) if ae_config.use_L2andLAB \
        else rgb_transform(reduce_mean_and_std_normalize_images(y_dec, ae_config), ae_config)

    # Use the x patches as filters and calculate convolution on the transposed y images to get correlation
    cross_corr = L2_or_pearson_corr(q, r, ae_config) * mask

    ncc = cross_corr
    ncc_h = tf.shape(ncc)[1]
    ncc_w = tf.shape(ncc)[2]

    # Take the maximal NCC value for each pair of images (y image + x patch)
    ncc_reshaped = tf.reshape(ncc, [batch_size, ncc_h * ncc_w, -1])
    if ae_config.use_L2andLAB:
        extermum_ncc = tf.cast(tf.argmin(ncc_reshaped, axis=1), dtype='int32') # for l2
    else:
        extermum_ncc = tf.cast(tf.argmax(ncc_reshaped, axis=1), dtype='int32')
    row = extermum_ncc // ncc_w
    col = extermum_ncc % ncc_w

    if batch_size == 1:
        box_ind = tf.constant(0, shape=[x_patches.get_shape()[0]])
        crop_size = tf.constant([patch_size_h, patch_size_w])
        boxes = tf.transpose(tf.concat([tf.expand_dims(row[0]/H, axis=0), tf.expand_dims(col[0]/W, axis=0),
                                        tf.expand_dims((row[0]+patch_size_h)/H, axis=0),
                                        tf.expand_dims((col[0]+patch_size_w)/W, axis=0)], 0))
        y_patches = tf.image.crop_and_resize(y_images_orig, tf.cast(boxes, tf.float32), box_ind, crop_size)

    elif batch_size > 1:
        # Every kernel(x_patch) created a map with every y_img, need to take only the relevant y_patch for each x_patch
        y_patches = []
        for i in range(batch_size):
            j = i
            y_patch = y_images[i, row[i,j]:row[i,j]+patch_size_h, col[i,j]:col[i,j]+patch_size_w, :]
            y_patches.append(y_patch)

        y_patches = tf.stack(y_patches)

    return y_patches, ncc, extermum_ncc, q, r, row, col


def reduce_mean_and_std_normalize_images(in_images, ae_config):

    if ae_config.use_L2andLAB:
        norm_images = 2 * ((tf.clip_by_value(in_images, 0., 255.)/255.) - 0.5)
    else:
        # values from KITTI dataset:
        means = [93.70454143384742, 98.28243432206516, 94.84678088809876]
        variances = [73.56493292844912, 75.88547006820752, 76.74838442810665]

        means = tf.tile([means], [in_images.get_shape().as_list()[0], 1])
        variances = tf.tile([variances], [in_images.get_shape().as_list()[0], 1])

        # Calculate mean in width and height per patch:
        means = tf.expand_dims(tf.expand_dims(means, axis=1), axis=1)
        variances = tf.expand_dims(tf.expand_dims(variances, axis=1), axis=1)
        norm_images = tf.divide((in_images - means), variances)

    return norm_images


def L2_or_pearson_corr(x, y, ae_config):
    """This func calculate the Pearson Correlation Coefficient/L2 between a patch x and all patches in image y
    Formula: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    R =  numerator/ denominator.
    where:
    numerator = sum_i(xi*yi - y_mean*xi - x_mean*yi + x_mean*y_mean)
    denominator = sqrt( sum_i(xi^2 - 2xi*x_mean + x_mean^2)*sum_i(yi^2 - 2yi*y_mean + y_mean^2) )

    Input: tensor of patchs x and img y
    Output: map that each pixel in it, is Pearson correlation/L2 correlative for a patch between x and y
    """
    N, H, W, C = x.get_shape().as_list()
    patch_size = int(H * W * C)

    # sum_i(xi*yi)
    xy = tf.nn.conv2d(y, tf.transpose(x, perm=[1, 2, 3, 0]), strides=[1, 1, 1, 1], padding='VALID')

    x_square = tf.square(x)
    y_square = tf.square(y)

    kernel_sum = tf.constant(1.0, shape=[H, W, C, 1])

    # sum_i(xi^2) , sum_i(yi^2)
    sum_x_square = tf.reduce_sum(tf.reshape(x_square, [N, -1]), axis=1)
    sum_y_square = tf.nn.conv2d(y_square, kernel_sum, strides=[1, 1, 1, 1], padding='VALID')

    if ae_config.use_L2andLAB:
        out = sum_x_square - 2 * xy + sum_y_square  # l2 output

    else:
        x_mean = tf.reduce_mean(tf.reshape(x, [N,-1]), axis=1)
        kernel_mean = tf.constant(1.0/patch_size, shape=[H,W,C,1])
        y_mean = tf.nn.conv2d(y, kernel_mean, strides=[1, 1, 1, 1], padding='VALID')
        sum_x = tf.reduce_sum(tf.reshape(x, [N,-1]), axis=1)

        sum_y = tf.nn.conv2d(y, kernel_sum, strides=[1, 1, 1, 1], padding='VALID')

        # y_mean*sum_i(xi) , x_mean*sum_i(yi)
        y_mean_x = tf.multiply(y_mean, sum_x)
        x_mean_y = tf.multiply(sum_y, x_mean)

        # x_mean*y_mean
        xy_mean = tf.multiply(y_mean, x_mean)

        # x_mean*sum_i(xi) , y_mean*sum_i(yi)
        x_mean_x = tf.multiply(x_mean, sum_x)
        y_mean_y = tf.multiply(y_mean, sum_y)

        # x_mean^2 , y_mean^2
        x_mean_square = tf.square(x_mean)
        y_mean_square = tf.square(y_mean)

        numerator = xy - y_mean_x - x_mean_y + patch_size*xy_mean
        denominator_x = sum_x_square - 2*x_mean_x + patch_size * x_mean_square
        denominator_y = sum_y_square - 2*y_mean_y + patch_size * y_mean_square
        denominator = tf.multiply(denominator_y, denominator_x)

        out = tf.divide(numerator, tf.sqrt(denominator))  # Pearson output

    return out


def rgb_transform(x, ae_config):
    """This func gets an RGB img tensor (channel last) and transforms it to:
     H1 H2 H3 img
    H1=R+G , H2=R-G , H3= -0.5*(R+B)
    according to: https://pdfs.semanticscholar.org/8120/fa0a8c35e96c7312ab994caa2d47fceb5f85.pdf
    or to LAB"""

    if ae_config.use_L2andLAB:
        x_trans = rgb_to_lab(x)

    else:
        R, G, B = tf.split(x, [1, 1, 1], axis=3)
        H1 = R + G
        H2 = R - G
        H3 = 0.5*(R + B)
        x_trans = tf.concat([H1, H2, H3], axis=3)
    return x_trans


def rgb_to_lab(srgb):
    # based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
    with tf.name_scope('rgb_to_lab'):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        with tf.name_scope('srgb_to_xyz'):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope('xyz_to_cielab'):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimensions')

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def color_transform(x, y):
    """This func gets x patches tensor, y patches tensor and finds (a,b) for R,G,B such that x=ay+b"""

    means_x, _ = tf.nn.moments(x, axes=[1, 2])
    means_x = tf.expand_dims(tf.expand_dims(means_x, axis=1), axis=1)

    means_y, _ = tf.nn.moments(y, axes=[1, 2])
    means_y = tf.expand_dims(tf.expand_dims(means_y, axis=1), axis=1)

    y_out = y - means_y + means_x

    return tf.clip_by_value(y_out, 0., 255.)
