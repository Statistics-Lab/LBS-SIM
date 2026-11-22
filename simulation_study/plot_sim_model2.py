import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


for cens in [0.1, 0.3, 0.5]:
    for n in [200, 400, 600]:
        # locate the result directory produced by the corresponding simulation script
        result_dirs = glob.glob(os.path.join(os.path.dirname(__file__), 'results', f'model2,m=1000,n={n},mean_censor_rate=*'))
        if not result_dirs:
            raise FileNotFoundError(
                f'No result directory found for model2, n={n}. '
                'Run sim_model2_n{cens}_*.py first.')
        result_dir = sorted(result_dirs)[0]

        # data
        var_path = glob.glob(os.path.join(result_dir, 'model2_sim_variables*.xlsx'))[0]
        df = pd.read_excel(var_path, sheet_name = 'model2_variables')
        data = np.array(df)
        w = data[:,[2,3]]
        u = data[:,[6,7]]

        g_true = data[:,8]
        g_mean = data[:,9]
        alpha_true = data[:, 10]
        alpha_mean = data[:, 11]
        df = pd.read_excel(var_path, sheet_name = 'g_pre')
        g_pre_m = np.array(df)
        df = pd.read_excel(var_path, sheet_name='alpha_pre')
        alpha_pre_m = np.array(df)

        # parameters
        res_path = glob.glob(os.path.join(result_dir, 'model2_sim_results*.xlsx'))[0]
        df = pd.read_excel(res_path, sheet_name = 'estimates')
        data = np.array(df)
        beta_m = data[:,[2,3]]
        rho_m = data[:,[6,7]]
        df = pd.read_excel(res_path, sheet_name = 'evaluation_metrics')
        beta_true = np.array(df['β_true'].dropna())
        beta_mean = np.array(df['β_mean'].dropna())
        rho_true = np.array(df['ρ_true'].dropna())
        rho_mean = np.array(df['ρ_mean'].dropna())

        # plotting
        plt.rcParams['axes.unicode_minus'] = False   # avoid the minus-sign rendering issue when saving figures
        plt.rcParams['font.family'] = 'Arial'
        # plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.figure(dpi=600)


        fig1, ax = plt.subplots()
        m = 1000
        for fig_iter in range(m):
            u1 = np.sort(w @ beta_m[fig_iter].T).flatten()
            g_pre = g_pre_m[:,fig_iter]

            u2 = np.sort(w @ beta_mean.T)

            u3 = np.sort(w @ beta_true.T)

            # ax.set_title(f"model1, n={n}, censor_rate={censor_rate_true:.3f}")
            plt.xlabel('$\mathrm{W^T\hat{\\beta}}$')
            plt.ylabel('$\mathrm{\hat{g}(W^T\hat{\\beta})}$')
            ax.plot(u1, g_pre, color=(0.7, 0.7, 0.7))
            ax.plot(u2, g_mean, '--b')  # plot of the estimated mean
            ax.plot(u3, g_true, '-r')  # raw data plot

        lines, = plt.plot([], [], '-r', label='Real curve')
        lines2, = plt.plot([], [], '--b', label='Single index')
        plt.legend(loc='upper left', frameon=False)
        plt.tight_layout()
        plt.show()

        fig2, ax = plt.subplots()
        for fig_alpha_iter in range(m):
            u4 = np.sort(u @ rho_m[fig_alpha_iter].T).flatten()
            alpha_pre = alpha_pre_m[:,fig_alpha_iter]

            u5 = np.sort(u @ rho_mean.T)

            u6 = np.sort(u @ rho_true.T)

            plt.xlabel('$\mathrm{U^T\hat{\\rho}}$')
            plt.ylabel('$\mathrm{\hat{\\psi}(U^T\hat{\\rho})}$')

            ax.plot(u4, alpha_pre, color=(0.7, 0.7, 0.7))
            ax.plot(u5, alpha_mean, '--b')
            ax.plot(u6, alpha_true, '-r')
        lines, = plt.plot([], [], '-r', label='Real curve')
        lines2, = plt.plot([], [], '--b', label='Single index')
        plt.legend(loc='upper left', frameon=False)
        plt.tight_layout()
        plt.show()


        # save figures
        folder_name = os.path.join(os.path.dirname(__file__), 'figures', 'sim_model2')
        os.makedirs(folder_name, exist_ok=True)

        fig1_path = os.path.join(folder_name, f'mean_cens={cens}_n={n}.png')
        fig1.savefig(fig1_path, dpi=600)
        fig2_path = os.path.join(folder_name, f'shape_cens={cens}_n={n}.png')
        fig2.savefig(fig2_path, dpi=600)