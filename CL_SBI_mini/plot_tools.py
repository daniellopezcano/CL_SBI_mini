import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def plot_dataset_Pk(dset_norm_mean, dset_norm_std, xx, list_model_names, len_models, colors, kk, plot_as_Pk=True):
    
    NN_plot = xx.shape[0]
    
    fig, axs = mpl.pyplot.subplots(2,1,figsize=(9,9), gridspec_kw={'height_ratios': [1.5, 1]})
    
    if plot_as_Pk:
        axs[0].set_ylabel(r'$P(k) \left[ \left(h^{-1} \mathrm{Mpc}\right)^{3} \right]$')
        axs[1].set_xlabel(r'$\mathrm{Wavenumber}\, k \left[ h\, \mathrm{Mpc}^{-1} \right]$')
        axs[1].set_ylabel(r'$P_\mathrm{Model}(k) / \left\langle P(k) \right\rangle$')
        xx_plot = 10**(xx*dset_norm_std + dset_norm_mean)
        # for kmax_plot in np.array([0.6, 0.2, -0.2, -0.6, -1.0, -1.4]):
            # axs[0].axvline(10**kmax_plot, c='k', ls=':', lw=1.)
            # axs[1].axvline(10**kmax_plot, c='k', ls=':', lw=1.)
    else:
        axs[0].set_ylabel(r'$\mathrm{Norm}\left(P(k) \left[ \left(h^{-1} \mathrm{Mpc}\right)^{3} \right]\right)$')
        axs[1].set_xlabel('$k - index [adim]$')
        axs[1].set_ylabel(r'$\frac{\mathrm{Norm}\left(P_\mathrm{Model}(k)\right)}{\mathrm{Norm}\left(P_\mathrm{mean\, , train}(k)\right)}$')
        N_kk = len(kk)
        kk = np.arange(N_kk)
        xx_plot = xx
        axs[0].axvline(N_kk, c='k', ls=':', lw=1.)
        axs[1].axvline(N_kk, c='k', ls=':', lw=1.)
        axs[0].axhline(0., c='k', lw=1, ls=':')

    linestyles = get_N_linestyles(NN_plot)
    ii_aug_column = 0
    custom_lines = []
    custom_labels = []
    custom_lines1 = []
    custom_labels1 = []
    for ii_model_dataset, len_model in enumerate(len_models):
        custom_lines.append(mpl.lines.Line2D([0],[0],color=colors[ii_model_dataset],ls='-',lw=10,marker=None,markersize=8))
        custom_labels.append(list_model_names[ii_model_dataset])
        for ii_cosmo in range(NN_plot):
            tmp_slice = slice(ii_aug_column, ii_aug_column+len_model)
            axs[0].plot(
                np.array(kk), xx_plot[ii_cosmo, tmp_slice].T,
                c=colors[ii_model_dataset], linestyle=linestyles[ii_cosmo], lw=1.5, marker=None, ms=2, alpha=0.7
            )
            axs[1].plot(
                np.array(kk), (xx_plot[ii_cosmo, tmp_slice]/np.mean(xx_plot[ii_cosmo], axis=0)).T,
                c=colors[ii_model_dataset], linestyle=linestyles[ii_cosmo], lw=1.5, marker=None, ms=2
            )         
            if (ii_model_dataset == 0):
                custom_lines1.append(mpl.lines.Line2D([0],[0],color='grey',ls=linestyles[ii_cosmo],lw=3,marker=None,markersize=8))
                custom_labels1.append("Cosmo #" + str(ii_cosmo))

        ii_aug_column += len_model
    
    legend = axs[0].legend(custom_lines, custom_labels, loc='upper right', fancybox=True, shadow=True, ncol=1,fontsize=14)
    axs[0].add_artist(legend)
    legend = axs[0].legend(custom_lines1, custom_labels1, loc='lower left', fancybox=True, shadow=True, ncol=2,fontsize=14)
    axs[0].add_artist(legend)
    
    axs[1].axhline(1., c='k', lw=1, ls=':')
    if plot_as_Pk:
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        # axs[0].set_xlim([0.01, 4.5])
        # axs[0].set_ylim([30., 70000.])
        axs[1].set_xscale('log')
        # axs[1].set_xlim([0.01, 4.5])
        # axs[1].set_ylim([0.8, 1.2])
    else:
        axs[0].set_xlim([0., len(kk)])
        axs[0].set_ylim([-2.5, 2.5])
        axs[1].set_xlim([0., len(kk)])
        axs[1].set_ylim([-10, 10])
    axs[0].set_xticklabels([])

    return fig, axs
    
