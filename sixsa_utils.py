import numpy as np
import torch
from pip._internal.utils.misc import tabulate
from sbi import utils as utils
from jaxspec.data.util import fakeit_for_multiple_parameters
from jaxspec.model.abc import SpectralModel
import time
from jaxspec.data import FoldingModel
import jax
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
from jaxspec.data import FoldingModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from datetime import datetime

def print_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = message.split('\n')
    formatted_message = f"[{timestamp}] [INFO] {lines[0]}"
    for line in lines[1:]:
        formatted_message += f"\n{' ' * (len(timestamp) + len('[INFO] ') + 2)}{line}"
    print(formatted_message)

def print_best_fit_parameters(free_parameter_names,free_parameter_prior_types,median,lower,upper,cstat,cstat_dev):

    # Apply transformation for "loguniform" prior types
    median[free_parameter_prior_types == "loguniform"] = 10. ** median[free_parameter_prior_types == "loguniform"]
    lower[free_parameter_prior_types == "loguniform"] = 10. ** lower[free_parameter_prior_types == "loguniform"]
    upper[free_parameter_prior_types == "loguniform"] = 10. ** upper[free_parameter_prior_types == "loguniform"]

    # Create a table using a loop
    table_data = [("Parameter", "Best fit", "Negative error", "Positive error")]

    for name , m , l , u in zip(free_parameter_names , median , lower , upper) :
        table_data.append((name , m , f"-{m - l:0.3f}" , f"+{u - m:0.3f}"))

    # Print the table
    print(tabulate(table_data , headers = "Results" , tablefmt = "fancy_grid"))
    print_message(f"Best fit c-stat={cstat:.3f}\nc-stat deviation-dev={cstat_dev:.3f}")

def generate_function_for_cmin_cmax_restrictor( cmin = 2000. , cmax = 5000. ) :
    def get_good_x( x ) :
        x_array_to_select = []
        n_bad = 0

        for x_p in x :
            good_or_bad_x = cmin <= np.sum(x_p.numpy( )) <= cmax
            n_bad += int(not good_or_bad_x)
            x_array_to_select.append(good_or_bad_x)
        fraction_good = 100. * (1. - n_bad / len(x_array_to_select))
        print(f"{cmin:.1f} {cmax:.1f} Number of simulations outside the range {n_bad:d} - "
              f"Number of good simulations {len(x_array_to_select) - n_bad:d} - "
              f"Good fraction = {fraction_good:.1f}%")

        return torch.as_tensor(x_array_to_select)

    return get_good_x


def compute_x_sim( jaxspec_model_expression , parameter_states , thetas , pha_file , energy_min , energy_max ,
                   free_parameter_prior_types , parameter_lower_bounds , apply_stat = True , verbose = False ) :

    #
    # Apply the transformation if needed
    #
    thetas = torch.as_tensor(np.where(np.array(free_parameter_prior_types) == "loguniform" , 10. ** thetas , thetas))

    jaxspec_model = SpectralModel.from_string(jaxspec_model_expression)

    parameter_values = []
    index_theta = 0

    for i_param , param_state in enumerate(parameter_states) :
        if param_state == "free" :
            parameter_values.append([thetas[j][index_theta] for j in range(len(thetas))])
            index_theta += 1
            if verbose :
                print(f"{param_state.lower( )} Parameter #{i_param + 1} of {jaxspec_model.n_parameters} ")

        elif param_state == "frozen" :
            parameter_values.append([parameter_lower_bounds[i_param] for j in range(len(thetas))])
            if verbose :
                print(f"{param_state.lower( )} Parameter #{i_param + 1} of {jaxspec_model.n_parameters} ")

    params_to_set = jaxspec_model.params
    i_para = 0

    for l , param_set in params_to_set.items( ) :
        for param_name , _ in param_set.items( ) :
            upd_dict = {param_name : np.array(parameter_values[i_para])}
            param_set.update(upd_dict)
            i_para += 1

    folding_model = FoldingModel.from_pha_file(pha_file , energy_min , energy_max)

    if len(thetas) > 1 :
        print("Multiple thetas simulated -> parallelization with JAX required")
        start_time = time.perf_counter( )
        x = jax.jit(lambda s : fakeit_for_multiple_parameters(folding_model , jaxspec_model , s ,
                                                              apply_stat = apply_stat))(params_to_set)

        end_time = time.perf_counter( )
        duration_time = end_time - start_time
        print(f"Run duration_time {duration_time:.1f} seconds for {len(thetas)} samples")
    #    return torch.as_tensor(np.array(x).astype(np.float32))
    else :
        print("One single theta simulated -> parallelization with JAX not required")
        x = fakeit_for_multiple_parameters(folding_model , jaxspec_model , params_to_set , apply_stat = apply_stat)
    return torch.as_tensor(np.array(x).astype(np.float32))

def compute_cstat( data_in: object , model_in: object , verbose: object = True ) -> object :
    from scipy.stats import norm
    import numpy

    #
    # From Kaastra(2017) https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..51K/abstract
    #

    def compute_ce_cv_from_kaastra_2017( mu ) :

        def f0( mu , k ) :
            import numpy as np
            import math
            #        print("before rounding,",mu,k)
            k = np.int32(k)
            pk_mu = (np.exp(-mu) * (mu ** k)) / math.factorial(k)
            if k > 0 :
                pk_mu = pk_mu * (mu - k + k * np.log(k / mu)) ** 2.
            if k == 0 :
                pk_mu = pk_mu * (mu) ** 2.

            return pk_mu

        import sys
        import numpy as np
        ce = 0.;
        cv = 0.

        if mu <= 0.5 : ce = -0.25 * mu ** 3. + 1.38 * mu ** 2. - 2. * mu * np.log(mu)
        if mu > 0.5 and mu <= 2. : ce = -0.00335 * mu ** 5 + 0.04259 * mu ** 4. - 0.27331 * mu ** 3. + 1.381 * mu ** 2. - 2. * mu * np.log(
            mu)
        if mu > 2 and mu <= 5. : ce = 1.019275 + 0.1345 * mu ** (0.461 - 0.9 * np.log(mu))
        if mu > 5 and mu <= 10. : ce = 1.00624 + 0.604 / mu ** 1.68
        if mu > 10 : ce = 1. + 0.1649 / mu + 0.226 / mu ** 2.

        if mu >= 0 and mu <= 0.1 : cv = 4. * (
                f0(mu , 0.) + f0(mu , 1.) + f0(mu , 2.) + f0(mu , 3.) + f0(mu , 4.)) - ce ** 2.
        if mu > 0.1 and mu <= 0.2 : cv = -262. * mu ** 4. + 195. * mu ** 3. - 51.24 * mu ** 2. + 4.34 * mu + 0.77005
        if mu > 0.2 and mu <= 0.3 : cv = 4.23 * mu ** 2. - 2.8254 * mu + 1.12522
        if mu > 0.3 and mu <= 0.5 : cv = -3.7 * mu ** 3. + 7.328 * mu ** 2 - 3.6926 * mu + 1.20641
        if mu > 0.5 and mu <= 1. : cv = 1.28 * mu ** 4. - 5.191 * mu ** 3 + 7.666 * mu ** 2. - 3.5446 * mu + 1.15431
        if mu > 1 and mu <= 2. : cv = 0.1125 * mu ** 4. - 0.641 * mu ** 3 + 0.859 * mu ** 2. + 1.0914 * mu - 0.05748
        if mu > 2 and mu <= 3. : cv = 0.089 * mu ** 3. - 0.872 * mu ** 2. + 2.8422 * mu - 0.67539
        if mu > 3 and mu <= 5. : cv = 2.12336 + 0.012202 * mu ** (5.717 - 2.6 * np.log(mu))
        if mu > 5 and mu <= 10. : cv = 2.05159 + 0.331 * mu ** (1.343 - np.log(mu))
        if mu > 10 : cv = 12. / mu ** 3. + 0.79 / mu ** 2. + 0.6747 / mu + 2.

        if ce == 0. or cv == 0. : sys.exit(
            "value of " + str(mu) + " not supported, please go back to Kaastra (2017)")
        #    print mu,ce,cv

        return ce , cv

    data = data_in.astype(numpy.float32)
    model = np.array(model_in).flatten( )
    #    print(np.shape(data))
    #    print(np.shape(model))

    if verbose : print("Total number of data bins=" , len(data))
    cstat = 0.
    ce_sum = 0.
    cv_sum = 0.
    chi2bfit = 0.
    for i in range(len(data)) :
        if model[i] <= 0 : model[i] = 1.0E-10
        if data[i] > 0. :  cstat += model[i] - data[i] - data[i] * np.log(model[i]) + data[i] * np.log(data[i])
        if data[i] <= 0. : cstat += model[i] - data[i] - data[i] * np.log(model[i]) + data[i]
        if data[i] > 0 : chi2bfit += ((data[i] - model[i]) ** 2) / data[i]
        ce , cv = compute_ce_cv_from_kaastra_2017(model[i])
        ce_sum += ce
        cv_sum += cv
    cstat = 2. * cstat
    if verbose : print(f"C-stat = {cstat:0.1f}")
    if verbose : print(f"Chi2  = {chi2bfit:0.1f}")
    if verbose : print(f"% Probability to get C-stat {cstat:0.1f} out of the expected C-stat {ce_sum:0.1f} "
                       f"with standard deviation {np.sqrt(cv_sum):0.1f} = {100. * norm.sf(np.abs((cstat - ce_sum) / np.sqrt(cv_sum))):0.1f}%"
                       f" - deviation ={(cstat - ce_sum) / np.sqrt(cv_sum):0.1f} sigma")

    return cstat , (cstat - ce_sum) / np.sqrt(cv_sum)

def plot_theta_in_theta_out( theta_test , posterior_samples_at_x_test , model_parameter_names , pdf_filename ) :
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
    median_array = [];
    median_em_array = [];
    median_ep_array = [];
    median_error_array = []
    for i_t in range(len(theta_test)) :
        median = np.median(posterior_samples_at_x_test[i_t] , axis = 0)
        lower , upper = np.percentile(posterior_samples_at_x_test[i_t] , (16 , 84) , axis = 0)
        median_array.append(median);
        median_em_array.append(median - lower);
        median_ep_array.append(upper - median)

        median_error_array.append(np.sqrt((median - lower) ** 2 + (upper - median) ** 2))

    fig , axs = plt.subplots(len(model_parameter_names) , 1 , figsize = (10 , 12))
    for iplot in range(len(model_parameter_names)) :
        xtp = [xtp[iplot] for xtp in theta_test]
        ytp = [ytp[iplot] for ytp in median_array]
        y_err = [1. / val[iplot] for val in median_error_array]
        xtp = np.array(xtp).reshape(-1 , 1);
        ytp = np.array(ytp).reshape(-1 , 1)
        print(np.shape(xtp) , np.shape(ytp) , np.shape(y_err))
        reg = LinearRegression( ).fit(np.array(xtp) , np.array(ytp) , sample_weight = np.array(y_err))
        R_2 = reg.score(np.array(xtp) , np.array(ytp) , sample_weight = np.array(y_err))
        wls = sm.WLS(np.array(ytp) , np.array(xtp) , weights = np.array(y_err))
        wls_result = wls.fit( )
        print(wls_result.summary( ))
        print(wls_result.params , R_2 , wls_result.conf_int(alpha = 0.1))
        print(dir(wls_result))
        axs[iplot].margins(x = 0.1)
        axs[iplot].margins(y = 0.1)
        local_label = model_parameter_names[iplot]
        if model_parameter_names[iplot].lower( ).find("norm") >= 0 :
            local_label = "Log(" + model_parameter_names[iplot] + ")"
        else :
            local_label = model_parameter_names[iplot]

        axs[iplot].set_xlabel("SRI " + local_label + " in")
        axs[iplot].set_ylabel("SRI " + local_label + " out")
        xtp = [xtp[iplot] for xtp in theta_test]
        ytp = [ytp[iplot] for ytp in median_array]
        axs[iplot].plot([np.min(xtp) , np.max(xtp)] ,
                        [wls_result.params[0] * np.min(xtp) , wls_result.params[0] * np.max(xtp)] , "--" ,
                        color = "green" , linewidth = 2. ,
                        label = f"Linear fit : a={wls_result.params[0]:0.4f} ($\pm$ {np.abs(wls_result.params[0] - np.array(wls_result.conf_int(alpha = 0.1))[0][0]):0.4f})")
        axs[iplot].errorbar(xtp , ytp ,
                            yerr = [[ytp[iplot] for ytp in median_em_array] , [ytp[iplot] for ytp in median_ep_array]] ,
                            fmt = 'o' , color = "red" , ms = 4 , mec = 'k' , ecolor = "k" , label = "Posterior median")

        axs[iplot].legend(frameon = False)
    fig.align_ylabels( )
    pdf.savefig(fig)
    matplotlib.pyplot.close( )
    pdf.close( )


def robust_selection_from_menu( menu ) :
    while True :

        for i in range(len(menu)) :
            print(f"{i + 1:d} --> {menu[i]}")
        # Ask the user to enter an integer between 1 and 10
        user_input = input(f"Please enter an integer between 1 and {len(menu):d}: ")

        # Try to convert the input to an integer
        try :
            user_integer = int(user_input)

            # Check if the integer is within the specified range
            if 1 <= user_integer <= len(menu) :
                print("You entered:" , user_integer)
                break  # Exit the loop if a valid integer is entered
            else :
                print(f"The integer should be between 1 and {len(menu):d}.")
        except ValueError :
            print("Please enter a valid integer.")
    return user_integer
