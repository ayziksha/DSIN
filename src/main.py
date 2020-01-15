import argparse
from decoder_imgcomp import decoder
from encoder_imgcomp import encoder
from siFull_img import SI_full_img
from siFinder import siFinder
from siNet import siNet as siNet
from AE import AE as AE
from tqdm import tqdm
import datetime
from utils import *
from DataProvider import *
import numpy as np
from fjcommon import config_parser
from lazyme.string import color_print
import os

""" This function calls the Train, Val and Test according to the user flags.
Input: config path to the ae_config and pc_config"""


def main(run_dict):

    val_phase_one, val_phase_two = False, False
    best_val, val_loss = np.Inf, np.Inf
    train_loss_history, val_loss_history, val_iters, train_iters = [], [], [], []
    train_sum, bpp_sum = 0.0, 0.0
    model_name = 'NA'
    iter = run_dict['total_iterations']
    now = datetime.datetime.today().strftime('%d%m%Y-%H%M')

    tf.reset_default_graph()
    cur_dir = os.getcwd() + '/data_paths/'
    ae = AE(run_dict['ae_config'], run_dict['pc_config'], encoder, decoder, siFinder, SI_full_img, siNet, cur_dir)
    data = Dataset(run_dict['ae_config'], cur_dir)

    val_names, test_names = data.get_data_size()
    val_iterations = len(val_names) // run_dict['batch_size']

    if run_dict['load_model']:
        model_name = run_dict['load_model_name']
        ae.load_model(run_dict['root_weights'] + model_name + '/model')
        iter = 'NA'
        lr = ('NA', 'NA')

    if run_dict['train_model']:
        lr = run_dict['ae_config'].lr_initial, run_dict['pc_config'].lr_initial
        tbar = tqdm(range(1, run_dict['total_iterations'] + 1))

        for iteration in tbar:
            data_np_train = data.get_data_for_train()

            x_train, y_train = data_np_train  # data_np_train is (x,y) tuple
            train_loss, bpp = ae.siNet_update(x_train, y_train)

            train_sum += train_loss
            bpp_sum += bpp

            if run_dict['decrease_val_steps']:
                """ reduce number of iteration between validation steps """
                run_dict['validate_every'], val_phase_one, val_phase_two = \
                    get_validate_every(iteration, run_dict['total_iterations'], run_dict['validate_every'],
                                       val_phase_one, val_phase_two)

            if iteration != 0 and iteration % run_dict['validate_every'] == 0:
                """ validation step """
                val_sum = 0
                for i in range(val_iterations):
                    data_np_val = data.get_data_for_val()
                    x_val, y_val = data_np_val  # data_np_val is (x,y) tuple
                    val_loss = ae.siNet_validate(x_val, y_val)
                    val_sum += val_loss

                val_loss = val_sum / float(val_iterations)
                val_loss_history.append(val_loss)
                val_iters.append(iteration)

                if val_loss < best_val:
                    best_val, iter = val_loss, iteration
                    if run_dict['save_model']:

                        model_name = save_model_fn(ae, run_dict['ae_config'], run_dict['pc_config'],
                                                   run_dict['root_weights'], now, iteration,
                                                   run_dict['total_iterations'], best_val, run_dict['save_config'])

            if iteration % run_dict['show_every'] == 0:
                """ print loss and iterations to console """
                bpp, train_loss_history, train_iters = print_to_console(run_dict['show_every'], train_sum,
                                                                        train_loss_history, train_iters, iteration,
                                                                        bpp_sum, val_loss, tbar)
                train_sum = 0.0
                bpp_sum = 0.0

        if run_dict['plot_loss_graph']:
            """ plot train + val loss_train (if train_model=True) """
            plot_loss(train_loss_history, val_loss_history, val_iters, train_iters, run_dict['total_iterations'],
                      best_val, iter, model_name)

            if run_dict['save_loss_graph']:
                plt.savefig(run_dict['root_save_img'] + 'images/loss_' + model_name + '.png')

    if run_dict['test_model']:

        for i in range(len(test_names)):
            print('Processing test image number {:d}'.format(i))
            data_np_test = data.get_data_for_test()

            x_test, y_test = data_np_test  # data_np_test is (x,y) tuple
            y_dec, y_syn, x_dec, x_with_si, bpp = ae.siNet_get_reconstructed(x_test, y_test)
            x_dec = np.clip(x_dec, 0, 255)
            x_with_si = np.clip(x_with_si, 0, 255)

            img_index = 0  # test image index to present within the batch
            if run_dict['plot_test_img']:
                """ show orig + decoded images """
                plot_inference(x_test[img_index], x_dec[img_index], y_test[img_index], y_syn[img_index],
                               x_with_si[img_index], model_name, run_dict['total_iterations'], iter, lr, bpp)

            if run_dict['save_test_img']:
                save_test_imgs_fn(run_dict['root_save_img'], model_name, x_with_si[img_index], i, bpp)

            if run_dict['create_loss_list']:
                x_rec = x_with_si
                if np.average(x_rec[img_index]) == 0:  # if no si (i.e. x_si will be zero) --> use x_dec
                    x_rec = x_dec
                loss_list_saver(x_test, y_test, x_rec, y_syn, run_dict['batch_size'], str(model_name), bpp,
                                run_dict['root_save_img'])


def get_validate_every(iteration, total_iterations, validate_every, val_phase_one, val_phase_two):

    if iteration > (total_iterations // 2) and val_phase_one is False:
        validate_every = validate_every // 10
        val_phase_one = True
    if iteration > 3*(total_iterations // 4) and val_phase_two is False:
        validate_every = validate_every // 2
        val_phase_two = True

    return validate_every, val_phase_one, val_phase_two


def save_model_fn(ae, ae_config, pc_config, root_weights, now, iteration, total_iterations, best_val, save_config):

    target_bpp = ae_config.H_target/(64./ae_config.num_chan_bn)
    if ae_config.AE_only:
        model_config = '_AE_only_'
    else:
        model_config = '_sinet_'

    model_name = 'target_bpp' + str(target_bpp) + model_config + now
    ae.save_model(root_weights + model_name + '/model')
    color_print('Saving to:' + root_weights + model_name, color='yellow')

    f = open(root_weights + 'last_saved_' + model_name + '.txt', 'w+')
    f.write(root_weights + model_name +
            '\nlast saved iteration number: ' + str(iteration) + '/' + str(total_iterations) +
            '\nlast saved val loss: ' + str(best_val))
    f.close()

    if save_config and not os.path.exists(root_weights + 'configs_' + model_name + '.txt'):
        f = open(root_weights + 'configs_' + model_name + '.txt', 'a+')
        f.write('#  ae configs:\n' + str(ae_config))
        f.write('\n\n#  pc configs:\n' + str(pc_config))
        f.close()

    return model_name


def print_to_console(show_every, train_sum, train_loss_history, train_iters, iteration, bpp_sum, val_loss, tbar):

    train_loss = train_sum / float(show_every)
    train_loss_history.append(train_loss)
    train_iters.append(iteration)
    bpp = bpp_sum / float(show_every)
    s = "Loss: {:.4f}, bpp: {:.4f}".format(train_loss, bpp)
    tbar.set_description(s)
    s = " Validation Loss: {:.4f}".format(val_loss)
    color_print(s, color='green')

    return bpp, train_loss_history, train_iters


def get_run_params(current_directory):

    ae_config, ae_config_rel_path = config_parser.parse(args.ae_config_path)
    pc_config, pc_config_rel_path = config_parser.parse(args.pc_config_path)
    run_dict = {'ae_config': ae_config,
                'ae_config_rel_path': ae_config_rel_path,
                'pc_config': pc_config,
                'pc_config_rel_path': pc_config_rel_path,
                'total_iterations': ae_config.iterations,
                'batch_size': ae_config.batch_size,
                'root_weights': current_directory + '/weights/',
                'root_save_img': current_directory + '/images/',
                'show_every': ae_config.show_every,
                'validate_every': ae_config.validate_every,  # max number of iteration between validation steps
                'decrease_val_steps': ae_config.decrease_val_steps,
                'load_model_name': ae_config.load_model_name,
                'load_model': ae_config.load_model,
                'train_model': ae_config.train_model,
                'test_model': ae_config.test_model,
                'save_model': ae_config.save_model,
                'plot_test_img': False,
                'save_test_img': True,
                'plot_loss_graph': False,
                'save_loss_graph': False,
                'create_loss_list': False,
                'save_config': True,
                }

    return run_dict


current_directory = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('-ae_config', '-ae_configs', '--ae_config_path', '--ae_configs_path', type=str,
                    help='AE config file path',
                    default=current_directory + '/run_configs/ae_run_configs')
parser.add_argument('-pc_config', '-pc_configs', '--pc_config_path', '--pc_configs_path', type=str,
                    help='PC config file path',
                    default=current_directory + '/run_configs/pc_run_configs')
args = parser.parse_args()

if __name__ == '__main__':
    main(get_run_params(current_directory))

