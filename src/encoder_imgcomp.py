import tensorflow as tf


def encoder(x_train, ae_imgcomp, is_training=True):

    with tf.variable_scope('encoder_body'):
        enc_out_train = ae_imgcomp.encode(x_train, is_training=is_training)  # qbar is masked by the heatmap (in training)

        return enc_out_train

