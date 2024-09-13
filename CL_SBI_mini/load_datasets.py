"""
This module contains functions for loading generated datasets and defining dataloader structures
that can be used for drawing batches during training and other utilities

Functions:
- load_stored_data: Load stored data from specified path and model names.
- draw_indexes_augs: Draw random indexes for augmentations.

Classes:
- data_loader: Data loader class for loading and processing data.
"""

import os
import itertools
import numpy as np
import pickle
import datetime
# import ipdb
import logging


def load_stored_data(
    path_load, list_model_names, return_len_models=False, include_baryon_params=False,
    np_name_cosmos="cosmos.npy", np_name_xx="xx.npy", np_name_ext_augs="extended_aug_params.npy"
    ):
    """
    Load stored data from specified path and model names.
    
    Parameters
    ----------
    path_load : str
        Path to load the data from.
    list_model_names : list
        List of model names to load.
    return_len_models : bool, optional
        Whether to return the length of models, by default False.
    include_baryon_params : bool, optional
        Whether to include baryon parameters, by default False.
    
    Returns
    -------
    tuple
        Loaded data including theta, xx, and optionally aug_params and len_models.
    """
    logging.info('Loading stored data...')
    
    xx = []
    aug_params = []
    len_models = []
    for ii, model_name in enumerate(list_model_names):
        logging.info('Loading ' + model_name + '...')
        with open(os.path.join(path_load, model_name + '_' + np_name_cosmos), 'rb') as ff:
            tmp_theta = np.load(ff)
            if ii == 0:
                theta = tmp_theta
            else:
                assert np.sum(tmp_theta != theta) == 0, "All theta values must coincide for the different models!"
                theta = tmp_theta
            
        with open(os.path.join(path_load, model_name + '_' + np_name_xx), 'rb') as ff:
            loaded_xx = np.load(ff)
        if include_baryon_params:
            with open(os.path.join(path_load, model_name + '_' + np_name_ext_augs), 'rb') as ff:
                loaded_aug_params = np.load(ff)
            
        len_models.append(loaded_xx.shape[1])
        
        if ii == 0:
            xx = loaded_xx
            if include_baryon_params:
                aug_params = loaded_aug_params
        else:
            xx = np.concatenate((xx, loaded_xx), axis=1)
            if include_baryon_params:
                aug_params = np.concatenate((aug_params, loaded_aug_params), axis=1)
        
    if include_baryon_params:
        if return_len_models:
            return theta, xx, aug_params, np.array(len_models)
        else:
            return theta, xx, aug_params
    else:
        if return_len_models:
            return theta, xx, np.array(len_models)
        else:
            return theta, xx


def func_add_noise_Pk(xx, NN_noise_realizations, kmax, box, factor_kmin_cut, gaussian_error_counter_tolerance=10):

    tmp_pk = 10**xx

    kf = 2.0 * np.pi / box
    kmin = np.log10(factor_kmin_cut * kf)
    N_kk = int((kmax - kmin) / (8 * kf))
    kk_log = np.logspace(kmin, kmax, num=N_kk)
    delta_log10kk = (np.log10(kk_log[1]) - np.log10(kk_log[0])) / 2
    kk_edges_log = 10**np.append(np.log10(kk_log) - delta_log10kk, np.log10(kk_log[-1]) + delta_log10kk)
    delta_kk = np.diff(kk_edges_log)
    
    cosmic_var_gauss_err = np.sqrt((4 * np.pi**2) / (box**3 * kk_log**2 * delta_kk)) * tmp_pk

    valid_sample = False
    while_counter = 0
    while not valid_sample:
        samples_pk = np.random.normal(loc=tmp_pk, scale=cosmic_var_gauss_err, size=(NN_noise_realizations,)+tmp_pk.shape)
        if np.sum(samples_pk < 0) == 0:
            valid_sample = True
        else:
            while_counter += 1
            tmp_indexes = np.where(samples_pk < 0)
            logging.warning(f"WARNING ({while_counter} / {gaussian_error_counter_tolerance}): gaussian error approximation failed. "
                            f"# Corrupted samples = {len(tmp_indexes[0])} / {samples_pk.shape[0] * samples_pk.shape[0]}")
            if while_counter > gaussian_error_counter_tolerance:
                import matplotlib as mpl
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                ax.set_title(f'# Corrupted samples = {len(tmp_indexes[0])} / {samples_pk.shape[0] * samples_pk.shape[0]}', fontsize=16)
                ax.set_xlabel(r'$\mathrm{Wavenumber}\, k \left[ h\, \mathrm{Mpc}^{-1} \right]$')
                ax.set_ylabel(r'$P(k) \left[ \left(h^{-1} \mathrm{Mpc}\right)^{3} \right]$')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.plot(kk_log, samples_pk[tmp_indexes[0], tmp_indexes[1]].T, c='k', alpha=0.9, marker=None, lw=0.5, ms=2)
                fig.tight_layout()
                fig.savefig('./gaussian_error_approximation_failed.png')
                assert False, f"ERROR: gaussian error approximation failed!!. Cosmology indexes: {tmp_indexes[0]}. Augmentation indexes: {tmp_indexes[1]}"
    
    xx = np.log10(samples_pk)
    
    return xx


def dset_normalization(xx, normalize, path_save_norm, path_load_norm, np_name_mean="mean.npy", np_name_std="std.npy"):
    
    if normalize and (path_save_norm is not None) and (path_load_norm is None):
        logging.info('Normalizing data and saving normalization parameters...')
        tmp_xx = np.reshape(xx, tuple([xx.shape[0]*xx.shape[1]*xx.shape[2],] + list(xx.shape[3:])))
        tmp_mean = np.mean(tmp_xx, axis=0)
        tmp_std = np.std(tmp_xx, axis=0)
        if not os.path.exists(path_save_norm):
            os.makedirs(path_save_norm)
        np.save(os.path.join(path_save_norm, np_name_mean), tmp_mean)
        np.save(os.path.join(path_save_norm, np_name_std), tmp_std)
        norm_mean = tmp_mean
        norm_std = tmp_std
        
    elif normalize and (path_load_norm is not None) and (path_save_norm is None):
        logging.info('Loading normalization parameters...')
        norm_mean = np.load(os.path.join(path_load_norm, np_name_mean))
        norm_std = np.load(os.path.join(path_load_norm, np_name_std))
        
    else:
        norm_mean = 0.
        norm_std = 1.
        
    xx = (xx - norm_mean) / norm_std
    
    return xx, norm_mean, norm_std


def concat_and_squeeze_augs_theta_and_xx(theta, xx, aug_params):
    theta = np.tile(theta[:, :, np.newaxis], (1, 1, xx.shape[2], 1))
    if aug_params is not None:
        theta = np.concatenate((theta, aug_params), axis=-1)
    theta = np.reshape(theta, (np.prod(theta.shape[0:-1]), theta.shape[-1]))
    xx = np.reshape(xx, (xx.shape[0]*xx.shape[1]*xx.shape[2], xx.shape[-1]))
    return theta, xx
    
    
def load_and_preprocess_dset(path_load_dset, list_model_names, dset_dict, normalize, path_save_norm, path_load_norm, dset_type="baccoemu", from_latents=False, include_baryon_params=False):
    
    np_name_xx="xx"
    if from_latents:
        np_name_xx += "_latents"
    np_name_xx += ".npy"

    theta, xx, aug_params, len_models = load_stored_data(
        path_load_dset,
        list_model_names,
        return_len_models=True,
        include_baryon_params=True,
        np_name_xx=np_name_xx
    )

    if dset_type == "baccoemu":
        add_noise_Pk          = dset_dict["add_noise_Pk"]
        NN_noise_realizations = dset_dict["NN_noise_realizations"]
        kmax                  = dset_dict["kmax"]
        box                   = dset_dict["box"]
        factor_kmin_cut       = dset_dict["factor_kmin_cut"]

        if add_noise_Pk == "cosmic_var_gauss":
            xx = func_add_noise_Pk(
                xx, NN_noise_realizations=NN_noise_realizations, kmax=kmax, box=box, factor_kmin_cut=factor_kmin_cut
            )
            theta = np.tile(theta[np.newaxis], (NN_noise_realizations, 1, 1))
            aug_params = np.tile(aug_params[np.newaxis], (NN_noise_realizations, 1, 1, 1))
        else:
            xx = xx[np.newaxis]
            theta = theta[np.newaxis]
            aug_params = aug_params[np.newaxis]

    else:
        logging.error('Unknown dset_type: %s', dset_type)
        raise ValueError(f"Unknown dset_type: {dset_type}")
    
    xx, norm_mean, norm_std = dset_normalization(
        xx, normalize=normalize, path_save_norm=path_save_norm, path_load_norm=path_load_norm
    )
    
    if not include_baryon_params:
        aug_params = None
    theta, xx = concat_and_squeeze_augs_theta_and_xx(theta, xx, aug_params)
    
    return theta, xx