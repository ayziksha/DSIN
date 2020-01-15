import tensorflow as tf
import ms_ssim_imgcomp as ms_ssim
from fjcommon import tf_helpers
import training_helpers_imgcomp as training_helpers


class Distortions(object):
    def __init__(self, config, x, x_out, is_training):
        assert tf.float32.is_compatible_with(x.dtype) and tf.float32.is_compatible_with(x_out.dtype)
        self.config = config

        with tf.name_scope('distortions_train' if is_training else 'distortions_test'):
            minimize_for = config.distortion_to_minimize
            assert minimize_for in ('mae', 'mse', 'psnr', 'ms_ssim')
            # don't calculate MS-SSIM if not necessary to speed things up
            should_get_ms_ssim = minimize_for == 'ms_ssim'
            # if we don't minimize for PSNR, cast x and x_out to int before calculating the PSNR, because otherwise
            # PSNR is off. If not training, always cast to int, because we don't need the gradients.
            # equivalent for when we don't minimize for MSE
            cast_to_int_for_psnr = (not is_training) or minimize_for != 'psnr'
            cast_to_int_for_mse = (not is_training) or minimize_for != 'mse'
            cast_to_int_for_mae = (not is_training) or minimize_for != 'mae'
            self.mae = self.mean_over_batch(
                Distortions.get_mae_per_img(x, x_out, cast_to_int_for_mae), name='mae')
            self.mse = self.mean_over_batch(
                Distortions.get_mse_per_img(x, x_out, cast_to_int_for_mse), name='mse')
            self.psnr = self.mean_over_batch(
                Distortions.get_psnr_per_image(x, x_out, cast_to_int_for_psnr), name='psnr')
            self.ms_ssim = (
                Distortions.get_ms_ssim(x, x_out)
                if should_get_ms_ssim else None)

            with tf.name_scope('distortion_to_minimize'):
                self.d_loss_scaled = self._get_distortion_to_minimize(minimize_for)

    def summaries_with_prefix(self, prefix):
        return tf_helpers.list_without_None(
            tf.summary.scalar(prefix + '/mae', self.mae),
            tf.summary.scalar(prefix + '/mse', self.mse),
            tf.summary.scalar(prefix + '/psnr', self.psnr),
            tf.summary.scalar(prefix + '/ms_ssim', self.ms_ssim) if self.ms_ssim is not None else None)

    def _get_distortion_to_minimize(self, minimize_for):
        """ Returns a float32 that should be minimized in training. For PSNR and MS-SSIM, which increase for a
        decrease in distortion, a suitable factor is added. """
        if minimize_for == 'mae':
            return self.mae
        if minimize_for == 'mse':
            return self.mse
        if minimize_for == 'psnr':
            return self.config.K_psnr - self.psnr
        if minimize_for == 'ms_ssim':
            return self.config.K_ms_ssim * (1 - self.ms_ssim)

        raise ValueError('Invalid: {}'.format(minimize_for))

    @staticmethod
    def mean_over_batch(d, name):
        assert len(d.shape) == 1, 'Expected tensor of shape (N,), got {}'.format(d.shape)
        with tf.name_scope('mean_' + name):
            return tf.reduce_mean(d, name='mean')

    @staticmethod
    def get_mae_per_img(inp, otp, cast_to_int):
        """
        :param inp: NCHW
        :param otp: NCHW
        :param cast_to_int: if True, both inp and otp are casted to int32 before the error is calculated,
        to ensure real world errors (image pixels are always quantized). But the error is always casted back to
        float32 before a mean per image is calculated and returned
        :return: float32 tensor of shape (N,)
        """
        with tf.name_scope('mae_{}'.format('int' if cast_to_int else 'float')):
            if cast_to_int:
                # Values are expected to be in 0...255, i.e., uint8, but tf.square does not support uint8's
                inp, otp = tf.cast(inp, tf.int32), tf.cast(otp, tf.int32)
            abs_error = tf.abs(otp - inp)
            abs_error_float = tf.to_float(abs_error)
            mae_per_image = tf.reduce_mean(abs_error_float, axis=[1, 2, 3])
            return mae_per_image

    @staticmethod
    def get_mse_per_img(inp, otp, cast_to_int):
        """
        :param inp: NCHW
        :param otp: NCHW
        :param cast_to_int: if True, both inp and otp are casted to int32 before the error is calculated,
        to ensure real world errors (image pixels are always quantized). But the error is always casted back to
        float32 before a mean per image is calculated and returned
        :return: float32 tensor of shape (N,)
        """
        with tf.name_scope('mse_{}'.format('int' if cast_to_int else 'float')):
            if cast_to_int:
                # Values are expected to be in 0...255, i.e., uint8, but tf.square does not support uint8's
                inp, otp = tf.cast(inp, tf.int32), tf.cast(otp, tf.int32)
            squared_error = tf.square(otp - inp)
            squared_error_float = tf.to_float(squared_error)
            mse_per_image = tf.reduce_mean(squared_error_float, axis=[1, 2, 3])
            return mse_per_image

    @staticmethod
    def get_psnr_per_image(inp, otp, cast_to_int):
        with tf.name_scope('psnr_{}'.format('int' if cast_to_int else 'float')):
            mse_per_image = Distortions.get_mse_per_img(inp, otp, cast_to_int)
            psnr_per_image = 10 * tf_helpers.log10(255.0 * 255.0 / mse_per_image)
            return psnr_per_image

    @staticmethod
    def get_ms_ssim(inp, otp):
        with tf.name_scope('mean_MS_SSIM'):
            return ms_ssim.MultiScaleSSIM(inp, otp, data_format='NCHW', name='MS-SSIM')

def get_loss(config, ae, pc, d_loss_scaled, bc, heatmap):
    assert config.H_target

    heatmap_enabled = heatmap is not None

    with tf.name_scope('losses'):
        bc_mask = (bc * heatmap) if heatmap_enabled else bc
        H_real = tf.reduce_mean(bc, name='H_real')
        H_mask = tf.reduce_mean(bc_mask, name='H_mask')
        H_soft = 0.5 * (H_mask + H_real)

        H_target = tf.constant(config.H_target, tf.float32, name='H_target')
        beta = tf.constant(config.beta, tf.float32, name='beta')

        pc_loss = beta * tf.maximum(H_soft - H_target, 0)

        # Adding Regularizers
        with tf.name_scope('regularization_losses'):
            reg_probclass = pc.regularization_loss()
            if reg_probclass is None:
                reg_probclass = 0
            reg_enc = ae.encoder_regularization_loss()
            reg_dec = ae.decoder_regularization_loss()
            reg_loss = reg_probclass + reg_enc + reg_dec

        pc_comps = [('H_mask',  H_mask),
                    ('H_real',  H_real),
                    ('pc_loss', pc_loss),
                    ('reg',     reg_probclass)]
        ae_comps = [('d_loss_scaled',     d_loss_scaled),
                    ('reg_enc_dec',       reg_enc + reg_dec)]

        total_loss = d_loss_scaled + pc_loss + reg_loss
        return total_loss, H_real, pc_comps, ae_comps
