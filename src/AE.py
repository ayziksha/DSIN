import numpy as np
import autoencoder_imgcomp as autoencoder
import probclass_imgcomp as probclass
import bits_imgcomp as bits
from Distortions_imgcomp import *

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)


class AE(object):

    def __init__(self, ae_config, pc_config, encoder, decoder, siFinder, SI_full_img, siNet, cur_dir):
        self.ae_config = ae_config
        self.pc_config = pc_config
        self._encode = encoder
        self._decode = decoder
        self.AE_only = self.ae_config.AE_only
        if self.AE_only:
            self.si_weight = 0.0
        else:
            self.si_weight = self.ae_config.si_weight
        self._siNet = siNet
        self._SI_full_img = SI_full_img
        self._siFinder = siFinder
        self.use_y_gauss_mask = self.ae_config.use_gauss_mask
        self._batch_size = self.ae_config.batch_size if self.AE_only else 1
        self._input_dim_h, self._input_dim_w = self.ae_config.crop_size
        self._y_patch_h, self._y_patch_w = self.ae_config.y_patch_size
        self.num_training_imgs = sum(1 for line in open(cur_dir + ae_config.file_path_train)) // 2

        ae_cls = autoencoder.get_network_cls(self.ae_config)
        pc_cls = probclass.get_network_cls(self.pc_config)

        # Initiate autoencoder and probability classifier
        self.ae_imgcomp = ae_cls(self.ae_config)
        self.pc_imgcomp = pc_cls(self.pc_config, num_centers=self.ae_config.num_centers)

        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('ae'):
            self.x = tf.placeholder(tf.float32, shape=(self._batch_size, 3, self._input_dim_h, self._input_dim_w),
                                    name='x_placeholder')
            self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
            self.side_info_placeholder = tf.placeholder(tf.float32, shape=(self._batch_size, 3, None, None),
                                                        name='side_info_placeholder')
            self.y_dec = tf.placeholder(tf.float32, shape=(self._batch_size, 3, None, None), name='y_dec_placeholder')
            self.mask = tf.constant(self.create_gaussian_masks()) if self.use_y_gauss_mask else 1

        with tf.variable_scope('encoder'):
            with tf.variable_scope("encoder_body", reuse=False):
                self.z = self._encode(self.x, self.ae_imgcomp, is_training=self.is_training)

        with tf.variable_scope('decoder'):
            self.x_dec = self._decode(self.z.qbar, self.ae_imgcomp, is_training=self.is_training)

        with tf.variable_scope('siFinder'):
            self.y_syn, self.ncc, self.extermum_ncc, self.q, self.r, self.row, self.col, self.x_patches, self.y_patches = self._SI_full_img(
                self.x_dec, self.side_info_placeholder, self.mask, self._y_patch_h, self._y_patch_w, self.ae_config,
                self.y_dec)

        with tf.variable_scope('siNetwork'):
            if self.AE_only:
                self.x_with_si = tf.zeros_like(self.x)
            else:
                concat = tf.concat([self.normalize(self.x_dec), tf.stop_gradient(self.normalize(self.y_syn))],
                                   axis=1)  # [N,6,H,W]
                self.x_with_si = self.denormalize(self._siNet(concat))  # output shape [N,3,H,W]

        with tf.variable_scope('imgcomp'):
            # Train part:
            # stop_gradient is beneficial for training. it prevents multiple gradients flowing into the heatmap.
            pc_in = tf.stop_gradient(self.z.qbar)
            bc_train = self.pc_imgcomp.bitcost(pc_in, self.z.symbols, is_training=self.is_training,
                                               pad_value=self.pc_imgcomp.auto_pad_value(self.ae_imgcomp))
            self.bpp_train = bits.bitcost_to_bpp(bc_train, self.x)
            d_train = Distortions(self.ae_config, self.x, self.x_dec, is_training=True)
            # loss_train ---
            self.total_loss, H_real, pc_comps, ae_comps = get_loss(
                self.ae_config, self.ae_imgcomp, self.pc_imgcomp, (1 - self.si_weight) * d_train.d_loss_scaled,
                bc_train, self.z.heatmap)

            # Test part:
            bc_test = self.pc_imgcomp.bitcost(self.z.qbar, self.z.symbols, is_training=self.is_training,
                                              pad_value=self.pc_imgcomp.auto_pad_value(self.ae_imgcomp))
            self.bpp_test = bits.bitcost_to_bpp(bc_test, self.x)
            # loss_test ---
            self.total_loss_test, H_real, pc_comps, ae_comps = get_loss(
                self.ae_config, self.ae_imgcomp, self.pc_imgcomp, (1 - self.si_weight) * d_train.d_loss_scaled,
                bc_test, self.z.heatmap)

        with tf.variable_scope('loss'):
            loss_siNet = tf.losses.absolute_difference(self.x, self.x_with_si) if not self.AE_only else 0
            if not self.AE_only and self.ae_config.batch_size > 1:
                self.loss_train = (self.total_loss + self.si_weight * loss_siNet) / float(self.ae_config.batch_size)
            else:
                self.loss_train = (self.total_loss + self.si_weight * loss_siNet)
            self.loss_test = self.total_loss_test + self.si_weight * loss_siNet

        with tf.variable_scope('training-step'):
            self.train_op_imgcomp, self.global_step = \
                self.get_train_op(self.ae_config, self.pc_config, self.pc_imgcomp.variables(), self.loss_train)

        self.sesh = tf.Session()
        self.sesh.run(tf.global_variables_initializer(), options=run_opts)

    def siNet_update(self, x, y):
        if not self.AE_only:
            y_dec = self.create_y_dec(y)
            _, _, loss, bpp = self.sesh.run([self.train_op_imgcomp, self.global_step, self.loss_train, self.bpp_train],
                                            feed_dict={self.x: x, self.side_info_placeholder: y, self.y_dec: y_dec,
                                                       self.is_training: True}, options=run_opts)
        else:
            _, _, loss, bpp = self.sesh.run([self.train_op_imgcomp, self.global_step, self.loss_train, self.bpp_train],
                                            feed_dict={self.x: x, self.side_info_placeholder: y,
                                                       self.is_training: True}, options=run_opts)
        return loss, bpp

    def siNet_validate(self, x, y):
        if not self.AE_only:
            y_dec = self.create_y_dec(y)
            loss = self.sesh.run(self.loss_test,
                                 feed_dict={self.x: x, self.side_info_placeholder: y, self.y_dec: y_dec,
                                            self.is_training: False}, options=run_opts)
        else:
            loss = self.sesh.run(self.loss_test,
                                 feed_dict={self.x: x, self.side_info_placeholder: y,
                                            self.is_training: False}, options=run_opts)
        return loss

    def siNet_get_reconstructed(self, x, y):
        if not self.AE_only:
            y_dec = self.create_y_dec(y)
            y_syn, x_dec, x_with_si, bpp, ncc, extermum_ncc, q, r, row, col, x_patches, y_patches = self.sesh.run(
                [self.y_syn, self.x_dec, self.x_with_si, self.bpp_test, self.ncc, self.extermum_ncc, self.q,
                 self.r,
                 self.row, self.col, self.x_patches, self.y_patches],
                feed_dict={self.x: x, self.side_info_placeholder: y, self.y_dec: y_dec,
                           self.is_training: False})
        else:
            y_dec = self.y_dec  # = None
            y_syn, x_dec, x_with_si, bpp, ncc, extermum_ncc, q, r, row, col, x_patches, y_patches = \
                self.sesh.run([self.y_syn, self.x_dec, self.x_with_si, self.bpp_test, self.ncc, self.extermum_ncc,
                               self.q, self.r, self.row, self.col, self.x_patches, self.y_patches],
                              feed_dict={self.x: x, self.side_info_placeholder: y, self.is_training: False})

        return y_dec, y_syn, x_dec, x_with_si, bpp

    def create_y_dec(self, y):
        y_dec = self.sesh.run([self.x_dec], feed_dict={self.x: y, self.is_training: False})
        return y_dec[0]

    def save_model(self, save_path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(self.sesh, save_path)

    def load_model(self, load_path):
        variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder/encoder_body') +\
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder') +\
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='imgcomp')  # do AE inference / train si 1st time

        if self.ae_config.load_train_step:  # train AE again
            variable_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training-step')

            if not self.AE_only:  # train si again
                variable_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='siNetwork')

        elif self.ae_config.test_model and not self.ae_config.train_model and not self.AE_only:  # do si inference
            variable_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='siNetwork')

        saver = tf.train.Saver(var_list=variable_list)

        print('Loading ' + load_path)
        saver.restore(self.sesh, load_path)

    def get_train_op(self, ae_config, pc_config, vars_probclass, total_loss):
        lr_ae, global_step = training_helpers.create_learning_rate_tensor(ae_config, ae_config.num_crops_per_img,
                                                                          self.num_training_imgs, self._batch_size,
                                                                          self.AE_only, name='lr_ae')
        default_optimizer = training_helpers.create_optimizer(ae_config, lr_ae, name='Adam_AE')

        lr_pc, _ = training_helpers.create_learning_rate_tensor(pc_config, ae_config.num_crops_per_img,
                                                                self.num_training_imgs, self._batch_size,
                                                                self.AE_only, name='lr_pc')
        optimizer_pc = training_helpers.create_optimizer(pc_config, lr_pc, name='Adam_PC')

        special_optimizers_and_vars = [(optimizer_pc, vars_probclass)]

        return tf_helpers.create_train_op_with_different_lrs(
                total_loss, default_optimizer, special_optimizers_and_vars, summarize_gradients=False), global_step

    def create_gaussian_masks(self):
        """ Creates a set of gaussian maps, each gaussian centered in patch_x center """
        patch_area = self._y_patch_h * self._y_patch_w
        img_area = self._input_dim_w * self._input_dim_h
        num_patches = np.arange(0, img_area // patch_area)
        patch_img_w = self._input_dim_w / self._y_patch_w
        w = np.arange(0, self._input_dim_w, 1, float)
        h = np.arange(0, self._input_dim_h, 1, float)
        h = h[:, np.newaxis]

        # mu = there is a gaussian map centered in each x_patch center
        center_h = (num_patches // patch_img_w + 0.5) * self._y_patch_h
        center_w = ((num_patches % patch_img_w) + 0.5) * self._y_patch_w

        # gaussian std
        sigma_h = 0.5 * self._input_dim_h
        sigma_w = 0.5 * self._input_dim_w

        # create the gaussian maps
        cols_gauss = (w - center_w[:, np.newaxis])[:, np.newaxis, :] ** 2 / sigma_w ** 2
        rows_gauss = np.transpose(h - center_h)[:,:, np.newaxis] ** 2 / sigma_h ** 2
        g = np.exp(-4 * np.log(2) * (rows_gauss + cols_gauss))

        # crop the masks to fit correlation map
        gauss_mask = g[:, self._y_patch_h // 2 - 1:self._input_dim_h - self._y_patch_h // 2,
                     self._y_patch_w // 2 - 1:self._input_dim_w - self._y_patch_w // 2]

        return np.transpose(gauss_mask.astype(np.float32), (1, 2, 0))[np.newaxis,:,:,:]

    def normalize(self, data):
        style = self.ae_config.normalization
        if style == 'OFF':
            return data
        if style == 'FIXED':
            mean, var = self.get_mean_var()
            return (data - mean) / np.sqrt(var + 1e-10)
        raise ValueError('Invalid normalization style {}'.format(style))

    def denormalize(self, data):
        style = self.ae_config.normalization
        if style == 'OFF':
            return data
        if style == 'FIXED':
            mean, var = self.get_mean_var()
            return (data * np.sqrt(var + 1e-10)) + mean
        raise ValueError('Invalid normalization style {}'.format(style))

    @staticmethod
    def get_mean_var():
        # values from KITTI dataset:
        mean = np.array([93.70454143384742, 98.28243432206516, 94.84678088809876], dtype=np.float32)
        var = np.array([5411.79935676, 5758.60456747, 5890.31451232], dtype=np.float32)

        # make mean, var into (3, 1, 1) so that they broadcast with NCHW
        mean = np.expand_dims(np.expand_dims(mean, -1), -1)
        var = np.expand_dims(np.expand_dims(var, -1), -1)

        return mean, var