import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from scipy.stats import linregress

class Helper:
    @staticmethod
    def plot_custom(df_list, x_param, param=None, y_param = 'Time', loglog=False, ax=None, label=' ', marker='o'):
        #x_param and label are the column names of things to plot Time vs X, for diff param
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        df_list = df_list if isinstance(df_list, list) else [df_list]

        for df in df_list:
            if param:
                labels = df[param].unique()
                if param == 'Lambda':
                    legend_label = 'λ'
                else: 
                    legend_label = param
                for val in labels:
                    subset = df[df[param] == val]
                    Helper.plot(ax, subset, x_param=x_param, param=param, y_param=y_param, loglog=loglog, label=f'{legend_label} = {val}', marker=marker)
                ax.legend()
            else:
                Helper.plot(ax, df, x_param=x_param, param=param, y_param=y_param, loglog=loglog, label=label, marker=marker)
        if label:    
            ax.set_title(f'{y_param} vs. {x_param} for {label}')
        else:
            ax.set_title(f'{y_param} vs. {x_param}')

        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.grid(True)

    # helper to plot_custom
    @staticmethod
    def plot(ax, data, x_param, y_param, param, loglog=False, label=None, marker='o'):
        if y_param == 'log_time':
            aux_x_vals = []
            aux_y_vals = []
            print("here")
            for idx, row in data.iterrows(): # iterates through the rows of the df
                logs = ast.literal_eval(row[y_param])
                aux_x_vals.extend([np.log(row[x_param])] * len(logs))
                aux_y_vals.extend(logs)
            ax.scatter(aux_x_vals, aux_y_vals, marker=marker, label=label, rasterized=True)
        
        elif loglog:
            line, = ax.loglog(data[x_param], data[y_param], marker=marker, linestyle=' ', label=label)
            color = line.get_color()

            # line of best fit in loglog
            log_values = Helper.get_loglog_values(data, x_param, y_param, param)
            a, b = log_values.iloc[0]['Alpha'], log_values.iloc[0]['Beta']            
            fit = np.exp(b)* (data[x_param].values)**a
            ax.loglog(data[x_param], fit, linestyle='--', color = color, label=f'a={a:.2f}')
        else:
            ax.plot(data[x_param], data[y_param], marker=marker, linestyle='-', label=label)
    
    def get_alpha_from_log(df, x_param='N', y_param='log_time', param='Lambda'):
        param_labels = df[param].unique()
        a_values = np.zeros(len(param_labels))
        b_values = np.zeros(len(param_labels))
        for i in range(len(param_labels)):
            data = df[df[param] == param_labels[i]]
            x = []
            y = []
            for _, row in data.iterrows():
                val = row[y_param]
                if isinstance(val, str):
                    logs = ast.literal_eval(val)
                elif isinstance(val, list):
                    logs = val
                else:
                    continue
                logs = [v for v in logs if np.isfinite(v)]
                x.extend([np.log(row[x_param])]*len(logs))
                y.extend(logs)
            x = np.array(x)
            y = np.array(y)

            a, b = np.polyfit(x, y, 1)
            a_values[i] = a
            b_values[i] = b

        log_transform_df = pd.DataFrame({
            param: param_labels,
            'Alpha': a_values,
            'Beta': b_values
        })

        return log_transform_df
    
    @staticmethod
    def same_y_axes(axes):
        y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])

        for ax in axes:
            ax.set_ylim(y_min, y_max)

    @staticmethod
    def get_loglog_values(df, x_param = 'N', y_param = 'Time', param = 'Lambda'):
        param_labels = df[param].unique()
        a_values = np.zeros(len(param_labels))
        b_values = np.zeros(len(param_labels))
        for i in range(len(param_labels)):
            data = df[df[param] == param_labels[i]]
            log_X = np.log(data[x_param])
            log_Y = np.log(data[y_param])
            a, b = np.polyfit(log_X, log_Y, 1) 
            a_values[i] = a
            b_values[i] = b

        log_transform_df = pd.DataFrame({
            param: param_labels,
            'Alpha': a_values,
            'Beta': b_values
        })

        return log_transform_df
    
    @staticmethod
    def make_df_list(df, param, param_values = None, param_symbol=None):
        if param_values is None:
            param_values = df[param].unique()
        if param_symbol is None:
            param_symbol = param
        data = []
        names = []
        for i in range(len(param_values)):
            data.append(df[df[param] == param_values[i]])
            names.append(param_symbol + '= ' + str(param_values[i]))
        return data, names

    @staticmethod
    def plot_param_vs_alpha(df_list, names = None, param='Lambda', label=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        df_list = df_list if isinstance(df_list, list) else [df_list]

        param_labels = df_list[0][param].unique()
        for i in range(len(df_list)):
            a_values = Helper.get_loglog_values(df_list[i])['Alpha']
            if names:
                ax.plot(param_labels, a_values, marker='o', linestyle='-', label=names[i])
            else:
                ax.plot(param_labels, a_values, marker='o', linestyle='-')
        ax.legend()
        ax.set_xlabel(param)
        ax.set_ylabel('Alpha')
        if label:
            ax.set_title(f'Alpha vs. {param} {label}')
        else:
            ax.set_title(f'Alpha vs. {param}')
        ax.grid(True)
    
    @staticmethod
    def save_results(results_time, results_energy, std_time, std_energy, name):
        N_values = [25, 40, 50, 100, 150, 200, 300]
        lambda_values = [1, 10, 25, 45, 70, 100]

        data = []

        for i in range(results_time.shape[0]):  # Loop over N values
            for j in range(results_time.shape[1]):  # Loop over lambda values
                data.append([
                    N_values[i],
                    lambda_values[j],
                    results_time[i, j],
                    results_energy[i, j], 
                    std_time[i, j],
                    std_energy[i, j]
                ])

        df = pd.DataFrame(data, columns=['N', 'Lambda', 'Time', 'Energy', 'std_Time', 'std_Energy'])
        df.to_csv(name + '.csv', index=False)
        return df

    # def layer_plot_simulations(ax, xs, yss, max_points=None, label=None, only_means=False, errorbars=False,
    #     color=None, sigma=1.0, fill_alpha=0.3, linestyle='-', linewidth=1, elinewidth=1):
    #     means = np.array([np.mean(ys) for ys in yss])
    #     devs = np.array([np.std(ys) * sigma for ys in yss])
    
    #     if max_points is not None:
    #         n = xs.shape[0]
    #         thin = max(1, n // max_points)
    
    #         xs = xs[::thin]
    #         means = means[::thin]
    #         devs = devs[::thin]
    
    #     if errorbars:
    #         ax.scatter(xs, means, c=color, label=label)
    #         ax.errorbar(xs, means, yerr=devs, linestyle=linestyle, linewidth=linewidth, elinewidth=elinewidth, c=color)
    #     else:
    #         ax.plot(xs, means, c=color, label=label, linestyle=linestyle, linewidth=linewidth)
    #         if not only_means:
    #             ax.fill_between(xs, means, means + devs, alpha=fill_alpha, color=color)
    #             ax.fill_between(xs, means, means - devs, alpha=fill_alpha, color=color)


    @staticmethod
    def layer_plot(ax, xs, means, devs, max_points=None,label=None, only_means=False, errorbars=False, color=None, 
        sigma=1.0, fill_alpha=0.3, linestyle='-', linewidth=1, elinewidth=1, capsize=3):
        # xs: x_vals, yss: list of lists (outcome for each x)
        devs = np.array([std*sigma for std in devs])
        if max_points is not None:
            n = xs.shape[0]
            thin = max(1, n // max_points)
    
            xs = xs[::thin]
            means = means[::thin]
            devs = devs[::thin]
    
        if errorbars:
            ax.scatter(xs, means, c=color, label=label)
            ax.errorbar(xs, means, yerr=devs, linestyle=linestyle, linewidth=linewidth, elinewidth=elinewidth, c=color, capsize=capsize)
        else:
            ax.plot(xs, means, c=color, label=label, linestyle=linestyle, linewidth=linewidth)
            if not only_means:
                ax.fill_between(xs, means, means + devs, alpha=fill_alpha, color=color)
                ax.fill_between(xs, means, means - devs, alpha=fill_alpha, color=color)

    @staticmethod
    def layer_plot_df(ax, df, x_param=None, y_param=None, y_devs=None, param=None, max_points=None,label=None, only_means=False, errorbars=False, color=None, 
        sigma=1.0, fill_alpha=0.3, linestyle='-', linewidth=1, elinewidth=1, capsize=3):
        # x_param, y_param, and param are strings, names for cols in df
        if param is None:
            Helper.layer_plot(ax, df[x_param], df[y_param], df[y_devs], max_points, label, only_means, errorbars, color, 
                sigma, fill_alpha, linestyle, linewidth, elinewidth, capsize)
        else:
            param_values = df[param].unique()

            for i in range(len(param_values)):
                data = df[df[param] == param_values[i]]
                Helper.layer_plot(ax, data[x_param], data[y_param], data[y_devs], max_points, label, only_means, errorbars, color, 
                    sigma, fill_alpha, linestyle, linewidth, elinewidth, capsize)
        
    @staticmethod
    def plot_hypercube_autocorrelation(sigma_trajectory, lags, ax = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        T = len(sigma_trajectory)
        N = len(sigma_trajectory[0])
        autocorrelations = {a: [] for a in lags}

        for t in range(max(lags), T):
            for a in lags:
                dot_product = np.dot(sigma_trajectory[t], sigma_trajectory[t - a]) / N
                autocorrelations[a].append(dot_product)

        plt.figure(figsize=(10, 6))
        for a, values in autocorrelations.items():
            ax.plot(range(max(lags), T), values, label=f'lag={a}')
        
        ax.set_xlabel('Time t')
        ax.set_ylabel('Autocorrelation ⟨sigma_t, sigma_(t-a)⟩')
        ax.legend()
        ax.set_title('Hypercube Autocorrelation')

##### Functions given log_time raw data

    @staticmethod
    def process_log_data(df):
        aux_x_vals = []
        aux_y_vals = []
        x_vals = []
        avg_y_vals = []
        for _, row in df.iterrows():
            log_time = row['log_time']
            if isinstance(log_time, str):
                log_time = ast.literal_eval(log_time)
            aux_x_vals.extend([np.log10(row['N'])] * len(log_time))
            aux_y_vals.extend(log_time)
            x_vals.append(np.log10(row['N']))
            avg_y_vals.append(np.mean(log_time))
        return np.array(aux_x_vals), np.array(aux_y_vals), x_vals, avg_y_vals

    # plot loglog; but also gets the alpha values for lambda
    def plot_log_time(df, ax, n=None, show=True, all_points = False, marker = 'o', title = 'loglog Plot Using log_time Data', show_error_bars = True, lambda_values = [1, 10, 25, 45, 70, 100], alpha_label = True):
        a_values = []
        se_values = []
        for lam in lambda_values:
            if n:
                data = df[(df['n'] == n) & (df['Lambda'] == lam)]
            else:
                data = df[df['Lambda']==lam]
            x_vals, y_vals, x, y = Helper.process_log_data(data)
            slope, intercept, _, _, std_err = linregress(x_vals, y_vals)
            a_values.append(slope)
            se_values.append(std_err)
            if show:
                if alpha_label:
                    ax.plot(x_vals, slope*np.array(x_vals) + intercept, linestyle='--', label=f'α={slope:.3f}')
                else:
                    ax.plot(x_vals, slope*np.array(x_vals) + intercept, linestyle='--')

                if not all_points:
                    if lam == -1:
                        name = r'$\infty$'
                    elif lam == 0:
                        name = '0'
                    else:
                        name = lam
                    ax.scatter(x, y, marker=marker, rasterized=True, label = f'λ = {name}')
                else:
                    ax.scatter(x_vals, y_vals, marker=marker, rasterized=True)
                ax.legend()
                ax.set_title(title, fontsize = 14)
                ax.set_xlabel(r'$\log N$', fontsize = 14)
                ax.set_ylabel(r'$\log t$', fontsize = 14)
                ax.grid()

            df_vals = pd.DataFrame({'log_N': x_vals, 'log_time': y_vals})
            grouped = df_vals.groupby('log_N')['log_time']
            log_N_vals = grouped.mean().index.values
            means = grouped.mean().values
            stds = grouped.std().values
            if show and show_error_bars:
                ax.errorbar(log_N_vals, means, yerr=stds, fmt=' ', capsize=5, color='gray')

        return np.array(a_values), np.array(se_values)
    
    def get_alpha_from_log_time(df, n=None):
        if n:
            data = df[df['n'] == n]
        else:
            data = df
        x_vals, y_vals, _, _ = Helper.process_log_data(data)
        slope, intercept, _, _, std_err = linregress(x_vals, y_vals)
        return slope, std_err

###### Extra plotting methods; better to use plot_custom() #######

    @staticmethod
    def plot_time(df, loglog=False):
        lambda_values = df['Lambda'].unique()
        for lambda_val in lambda_values:
            data = df[df['Lambda'] == lambda_val]

            if loglog:
                line, = plt.loglog(data['N'], data['Time'], marker='o', linestyle=' ', label=f'λ = {lambda_val}')
                color = line.get_color()

                # line of best fit in loglog
                log_N = np.log(data['N'])
                log_time = np.log(data['Time'])

                a, b = np.polyfit(log_N, log_time, 1) 
                fit = np.exp(b)* (data['N'].values)**a
                plt.loglog(data['N'], fit, linestyle='--', color = color, label=f'a={a:.2f}')
            else:
                plt.plot(data['N'], data['Time'], marker='o', linestyle='-', label=f'λ = {lambda_val}')
        
        plt.xlabel('N')
        plt.ylabel('Time')
        plt.title('Time vs. N for Different λ Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_energy(df):
        lambda_values = df['Lambda'].unique()
        for lambda_val in lambda_values:
            data = df[df['Lambda'] == lambda_val]
            plt.plot(data['N'], data['Energy'], marker='o', linestyle='-', label=f'λ = {lambda_val}')
        
        plt.xlabel('N')
        plt.ylabel('Energy')
        plt.title('Energy vs. N for Different λ Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_time_and_energy(df, loglog=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
        lambda_values = df['Lambda'].unique()
        for lambda_val in lambda_values:
            data = df[df['Lambda'] == lambda_val]
            if loglog:
                line, = ax1.loglog(data['N'], data['Time'], marker='o', linestyle=' ', label=f'λ = {lambda_val}')
                color = line.get_color()

                # line of best fit in loglog
                log_N = np.log(data['N'])
                log_time = np.log(data['Time'])

                a, b = np.polyfit(log_N, log_time, 1)
                fit = np.exp(b)* (data['N'].values)**a
                ax1.loglog(data['N'], fit, linestyle='--', color = color, label=f'a={a:.2f}') 
            else:
                ax1.plot(data['N'], data['Time'], marker='o', linestyle='-', label=f'λ = {lambda_val}')
        ax1.set_xlabel('N')
        ax1.set_ylabel('Time')
        ax1.set_title('Time vs. N for Different λ Values')
        ax1.legend()
        ax1.grid(True)

        for lambda_val in lambda_values:
            data = df[df['Lambda'] == lambda_val]
            ax2.plot(data['N'], data['Energy'], marker='o', linestyle='-', label=f'λ = {lambda_val}')

        ax2.set_xlabel('N')
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy vs. N for Different λ Values')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_energy_side_view(df1, df2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
        lambda_values = df1['Lambda'].unique()
        for lambda_val in lambda_values:
            data = df1[df1['Lambda'] == lambda_val]
            ax1.plot(data['N'], data['Energy'], marker='o', linestyle='-', label=f'λ = {lambda_val}')
        ax1.set_xlabel('N')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy vs. N for Different λ Values')
        ax1.legend()
        ax1.grid(True)

        for lambda_val in lambda_values:
            data = df2[df2['Lambda'] == lambda_val]
            ax2.plot(data['N'], data['Energy'], marker='o', linestyle='-', label=f'λ = {lambda_val}')

        ax2.set_xlabel('N')
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy vs. N for Different λ Values')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()