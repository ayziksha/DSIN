

def decoder(z,ae_imgcomp, is_training=True):

    x_out_train = ae_imgcomp.decode(z, is_training=is_training)

    return x_out_train
