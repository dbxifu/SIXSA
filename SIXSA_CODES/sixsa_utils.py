import time
from datetime import datetime

import jax
import ast
import sys
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import torch
from jaxspec.data import ObsConfiguration
from jaxspec.data.util import fakeit_for_multiple_parameters
from jaxspec.model.abc import SpectralModel
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

# Utility to print messages to the terminal in some format
def print_message(message):
    lines = message.split('\n')
    formatted_message="\n================================================================================\n"
    formatted_message+= '\n'.join("[SIXSA] " + line for line in lines)
    formatted_message+="\n================================================================================\n"
    print(formatted_message)

def welcome_message():
    print_message('\n\n\nWelcome in SIXSA (Simulation-based Inference for X-ray Spectral Analyis)\n'
                  'It works with us. Let us see how it does with you !\n'
                  'But remember according to the law of Murphy \n"Anything that can go wrong will go wrong."\n\n\n')

def goodbye_message():
    print_message("\n\n\nThank you for trying SIXSA (Simulation-based Inference for X-ray Spectral Analyis)."
          "\nWe hope you enjoyed it.\nNow you can customize it for your application."
          "\nKeep us posted (dbarret@irap.omp.eu, sdupourque@irap.omp.eu). Thanks !\n\n\n")
# Utility to print the best bit parameters in a tabulated form.
def print_best_fit_parameters(x_obs,free_parameter_names,free_parameter_prior_types,median,lower,upper,cstat,cstat_dev):

    # Apply transformation for "loguniform" prior types without copy
    median = torch.as_tensor(np.where(np.array(free_parameter_prior_types) == "loguniform" , 10. ** median , median))
    lower = torch.as_tensor(np.where(np.array(free_parameter_prior_types) == "loguniform" , 10. ** lower , lower))
    upper = torch.as_tensor(np.where(np.array(free_parameter_prior_types) == "loguniform" , 10. ** upper , upper))


    # Create a table using a loop
    table_data = [("Parameter", "Best fit", "Negative error", "Positive error")]

    for name , m , l , u in zip(free_parameter_names , median , lower , upper) :
        table_data.append((name , f"{m:0.3f}" , f"-{m - l:0.3f}" , f"+{u - m:0.3f}"))

    # Print the table
    print(tabulate(table_data, tablefmt = "fancy_grid"))
    print_message(f"These are the best fit results\nBest fit c-stat={cstat:.3f} ({len(x_obs)-len(free_parameter_names):d} d.o.f) - c-stat deviation={cstat_dev:.3f}")

#=======================================================================================================================
# generate_function_for_cmin_cmax_restrictor: this is the function that is needed for the restricted prior, derived from
# the condition that the spectra must have a number of counts within a specified range. Simulation falling outside the
# range are considered invalid and the network gets trained to increase the number of valid simulations.
# See sbi documentation for defining a restricted prior with a selection function.
# ======================================================================================================================

def generate_function_for_cmin_cmax_restrictor( cmin = 2000. , cmax = 5000. ) :

    def get_good_x( x ) :
        x_array_to_select = [cmin <= np.sum(x_p.numpy( )) <= cmax for x_p in x]
        n_bad = len(x) - sum(x_array_to_select)
        fraction_good = 100. * (1. - n_bad / len(x_array_to_select))
        print(f"{cmin:.1f} {cmax:.1f} Number of simulations outside the range {n_bad:d} - "
              f"Number of good simulations {len(x_array_to_select) - n_bad:d} - "
              f"Good fraction = {fraction_good:.1f}%")

        return torch.as_tensor(x_array_to_select)

    return get_good_x


# This function can be used to construct a restricted prior based on the simulated spectra providing the lowest cstat

def generate_function_for_cstat_restrictor( x_obs = [] , good_fraction_in_percent = 25. ) :
    def get_good_x( x ) :
        cstat_array = np.array(
            [compute_cstat(x_obs , x_p.numpy( ) , with_cstat_dev = False , verbose = False) for x_p in x])
        cstat_array_sorted = np.sort(cstat_array)

        index_corresponding_to_good_fraction = int(len(x) * good_fraction_in_percent / 100.)
        cstat_thresh = cstat_array_sorted[index_corresponding_to_good_fraction]
        print(f"cstat less than {cstat_thresh:0.1f} has {good_fraction_in_percent:0.1f}% of spectra at index "
              f"{index_corresponding_to_good_fraction:d} - median {np.median(cstat_array):0.1f} - "
              f"min {np.min(cstat_array):0.1f}")
        x_array_to_select=torch.as_tensor(cstat_array <= cstat_thresh)
#        for i in range(len(cstat_array)):
#            print(i,cstat_array[i],x_array_to_select[i])
#        input("Press Enter to continue")
#        return x_array_to_select
        return torch.as_tensor(cstat_array <= cstat_thresh)

    return get_good_x

def generate_cstat_good_fraction_restrictor_function(x_o=[],good_fraction_in_percent=10.) :

    def get_good_x(x):
        cstat_array=[]
        for x_p in x :
            cstat_array.append(compute_cstat(x_o , x_p.numpy( ) , with_cstat_dev = False , verbose = False))
        cstat_array_sorted=np.sort(cstat_array)
        index_corresponding_to_good_fraction=np.int32(len(x)*good_fraction_in_percent/100.)
        print(f"cstat less than {cstat_array_sorted[index_corresponding_to_good_fraction]:0.1f} has {good_fraction_in_percent:0.1f} \% of spectra at index {index_corresponding_to_good_fraction:d} - median {np.median(cstat_array):0.1f} - min {np.min(cstat_array):0.1f}")
        cstat_thresh=cstat_array_sorted[index_corresponding_to_good_fraction]
        good_x_array=[]
        for c in cstat_array :
            if c <= cstat_thresh :
                good_x_array.append(True)
            else :
                good_x_array.append(False)
#        print(cstat_array)
        return torch.as_tensor(good_x_array)
    return get_good_x

#=======================================================================================================================
# compute_x_sim: compute the simulated spectra with jaxspec fakeit like command.
# It is therefore dependent on jaxspec, which currently has a limited number of models implemented.
# It is possible to generate simulated spectra with other software, such as XSPEC, as long as the output format
# remains similar. The output format is an array of spectra in counts. jaxspec is really powerful in terms of speed.
# More models will be implemented as time goes (see the jaxspec documentation for the synthax of the models).
# ======================================================================================================================
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

    folding_model = ObsConfiguration.from_pha_file(pha_file , energy_min , energy_max)

    if len(thetas) > 1 :
        print("Multiple thetas simulated -> parallelization with JAX required")
        start_time = time.perf_counter( )
        x = jax.jit(lambda s : fakeit_for_multiple_parameters(folding_model , jaxspec_model , s ,
                                                              apply_stat = apply_stat))(params_to_set)

        end_time = time.perf_counter( )
        duration_time = end_time - start_time
        print(f"It took just {duration_time:.1f} seconds for jax.jit to generate {len(thetas)} simulations")
    #    return torch.as_tensor(np.array(x).astype(np.float32))
    else :
        print("One single theta simulated -> parallelization with JAX not required")
        x = fakeit_for_multiple_parameters(folding_model , jaxspec_model , params_to_set , apply_stat = apply_stat)
    return torch.as_tensor(np.array(x).astype(np.float32))

# This function computes the cstat, its expected value and variance.
def compute_cstat( data_in: object , model_in: object , with_cstat_dev=True, verbose: object = True ) -> object :
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
        if with_cstat_dev :
            ce , cv = compute_ce_cv_from_kaastra_2017(model[i])
            ce_sum += ce
            cv_sum += cv
    cstat = 2. * cstat
    if verbose : print(f"C-stat = {cstat:0.1f}")
    if verbose : print(f"Chi2  = {chi2bfit:0.1f}")
    if with_cstat_dev :
        if verbose : print(f"% Probability to get C-stat {cstat:0.1f} out of the expected C-stat {ce_sum:0.1f} "
                       f"with standard deviation {np.sqrt(cv_sum):0.1f} = {100. * norm.sf(np.abs((cstat - ce_sum) / np.sqrt(cv_sum))):0.1f}%"
                       f" - deviation ={(cstat - ce_sum) / np.sqrt(cv_sum):0.1f} sigma")

    if with_cstat_dev :
        return cstat , (cstat - ce_sum) / np.sqrt(cv_sum)
    else :
        return cstat

# This function plots the SIXSA inferred model parameters versus the input model parameters and compute some correlation
# coefficient which should be closed to 1.
def plot_theta_in_theta_out( theta_test , posterior_samples_at_x_test , model_parameter_names , pdf_filename ) :

    # update of the code to better handle the fits

    import numpy as np
    import matplotlib.pyplot as plt
    errors_upper = [];
    errors_lower = [];
    fitted_parameters = []
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

    for i_t in range(len(theta_test)) :
        median = np.median(posterior_samples_at_x_test[i_t] , axis = 0)
        lower , upper = np.percentile(posterior_samples_at_x_test[i_t] , (16 , 84) , axis = 0)
        fitted_parameters.append(median)
        errors_lower.append(median - lower)
        errors_upper.append(upper - median)

    degree = 1

    for iplot in range(len(model_parameter_names)) :
        xtp = np.array([val[iplot] for val in theta_test])
        ytp = np.array([val[iplot] for val in fitted_parameters])
        em_ytp = np.array([val[iplot] for val in errors_lower])
        ep_ytp = np.array([val[iplot] for val in errors_upper])

        # Calculate RMS of errors
        errors_rms = np.sqrt((em_ytp ** 2 + ep_ytp ** 2) / 2)

        # I am considering two ways of performing the linear regression - First way
        # Fit with LinearRegression
        model = LinearRegression(fit_intercept = False)
        model.fit(xtp.reshape(-1 , 1) , ytp , sample_weight = 1 / errors_rms ** 2)

        # Calculate the expected values (predicted values) using LinearRegression
        expected_values_linear_reg = model.predict(xtp.reshape(-1 , 1))

        # Calculate the residuals
        residuals_linear_reg = ytp - expected_values_linear_reg

        # Compute chi-square
        chi_square_linear_reg = np.sum((residuals_linear_reg / errors_rms) ** 2)

        # Express residuals in terms of sigma
        sigma_residuals_linear_reg = residuals_linear_reg / errors_rms
        print_message("Linear regression with sklearn.linear_model.LinearRegression")
        print(f'Linear Regression Coefficients with sklearn.linear_model.LinearRegression: {model.coef_}')
        print(f'Linear Regression Chi-square with sklearn.linear_model.LinearRegression: {chi_square_linear_reg:.4f}')
        print( )

        print_message("Linear regression with statsmodels.api.WLS (intercept frozen at 0)")
        # I am considering two ways of performing the linear regression - second way
        # Create a linear regression model with the intercept fixed to zero
        model = sm.WLS(ytp , xtp , weights = 1 / errors_rms ** 2 , hasconst = False)

        # Fit the model
        results = model.fit( )
        print(results.summary( ))
        # Get the predicted values
        predicted_values = results.fittedvalues
        # Access the 95% confidence interval for each coefficient
        confidence_interval = results.conf_int(alpha = 0.05)
        # Extract lower and upper bounds
        lower_bounds = confidence_interval[: , 0]
        upper_bounds = confidence_interval[: , 1]

        # Calculate positive and negative errors
        positive_errors = upper_bounds - results.params
        negative_errors = results.params - lower_bounds

        residuals = ytp - predicted_values

        # Calculate the chi-square value using regular residuals
        chi_square = np.sum((residuals / errors_rms) ** 2)

        # Degrees of freedom
        df = len(xtp) - results.df_model

        # Print the chi-square value and degrees of freedom
        print(f"statsmodels.api.WLS Chi-Square: {chi_square:.2f}, Degrees of Freedom: {df}")

        # Create two vertically stacked subplots with reduced vertical space
        fig , (ax1 , ax2) = plt.subplots(2 , 1 , sharex = True , gridspec_kw = {'height_ratios' : [3 , 1]})

        # Adjust the vertical space between subplots
        plt.subplots_adjust(hspace = 0)

        # Plot the original data with error bars in the top subplot
        ax1.errorbar(xtp , ytp , yerr = errors_rms , fmt = 'o' , color = "red" , ms = 4 , mec = 'k')
        ax1.margins(x = 0.1)
        ax1.margins(y = 0.1)
        # Plot the fitted line in the top subplot
        ax1.plot(xtp , results.predict(xtp) , color = 'black' , label = 'Linear regression')

        # Confidence interval using get_prediction
        pred = results.get_prediction( )
        confidence_interval = pred.conf_int(alpha = 0.005)
        # Plot the uncertainty region for the fitted line (shaded)
        ci_low , ci_high = confidence_interval.T

        sorted_indices = np.argsort(xtp)
        xtp_sorted = xtp[sorted_indices]
        ci_low_sorted = ci_low[sorted_indices]
        ci_high_sorted = ci_high[sorted_indices]

        # Plot the uncertainty region for the fitted line (shaded)
        ax1.fill_between(xtp_sorted , ci_low_sorted , ci_high_sorted , color = 'gray' , alpha = 0.3 ,
                         label = '95\% Confidence Interval')

        # Get the slope and its standard error
        slope = results.params[0]
        slope_error = results.bse[0]
        # Plot the ground truth line (without errors)
        ax1.plot([min(xtp) , max(xtp)] , [min(xtp) , max(xtp)] , linestyle = '--' , color = 'green' ,
                 label = 'Ground truth Line')

        # Annotate chi-square, degrees of freedom, slope, and slope error on the top subplot
        ax1.set_title(
            f'Linear coefficient = {slope:.3f} $\pm$ {np.mean([positive_errors[0] , negative_errors[0]]):.4f} \n$\chi^2$={chi_square:.1f} ({np.int32(df):d} d.o.f) ')

        ax1.set_ylabel('Fitted Parameters (' + model_parameter_names[iplot] + ")")
        ax1.legend( )

        # Plot the residuals in the bottom subplot
        ax2.errorbar(xtp , residuals / errors_rms , yerr = np.ones(len(xtp)) , fmt = 'o' , color = "red" , ms = 4 ,
                     mec = 'k')

        ax2.axhline(0 , color = 'black' , linestyle = '--' , linewidth = 1)  # Add horizontal line at y=0

        ax2.set_xlabel('Ground Truth (' + model_parameter_names[iplot] + ")")
        ax2.set_ylabel('Residuals ($\sigma$)')
        fig.align_ylabels( )
        pdf.savefig(fig)
        matplotlib.pyplot.close( )

        print_message("For checking : Linear regression with statsmodels.api.WLS (intercept left as a free parameter )")
        # I am considering two ways of performing the linear regression - second way
        # Create a linear regression model with the intercept fixed to zero
        xtp = sm.add_constant(xtp)
        model = sm.WLS(ytp , xtp , weights = 1 / errors_rms ** 2 , hasconst = True)

        # Fit the model
        results = model.fit( )
        print(results.summary( ))

    #        input("Press to continue")

    pdf.close( )


# Utility to robustly select an option from a menu
def robust_selection_from_menu(title, menu, return_index=True):
    # Display the title
    print_message(title)

    # Display the default selection
    default_selection = 1
    print(f"Default selection: {default_selection} --> {menu[default_selection - 1]} (default)")

    while True:
        # Display the menu options
        for i in range(len(menu)):
            if i == default_selection - 1:
                print(f"{i + 1:d} --> {menu[i]} (default)")
            else:
                print(f"{i + 1:d} --> {menu[i]}")

        # Ask the user to enter a selection
        user_input = input(f"Please enter a selection (1-{len(menu)}), or hit Enter for the default: ")

        # If the user hits return, return the default (1st option)
        if not user_input:
            print(f"You selected the default: {default_selection} --> {menu[default_selection - 1]}")
            return default_selection if return_index else menu[default_selection - 1]

        # Try to convert the input to an integer
        try:
            user_integer = int(user_input)

            # Check if the integer is within the specified range
            if 1 <= user_integer <= len(menu):
                print(f"You selected: {user_integer} --> {menu[user_integer - 1]}")
                return user_integer if return_index else menu[user_integer - 1]
            else:
                print(f"The selection should be between 1 and {len(menu)}.")
        except ValueError:
            print("Please enter a valid integer.")



def extract_and_print_imports():
    calling_frame = sys._getframe(1)
    file_path = calling_frame.f_code.co_filename

    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                print(f"Imported: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module is not None:
                for alias in node.names:
                    print(f"Imported: {module}.{alias.name}")


def summary_statistics_func(
    data: np.ndarray,
    energy_grid=None,
    energy_ref=None,
    with_basic_stats=True,
    with_sum=True,
    with_ratio=True,
    with_diff=True,
    with_energy_weighted=False,
):

    if data.ndim == 1:
        data = data[np.newaxis, :]  # (1, M)
    num_spectrum, num_bins = data.shape

    data_transformed_list = []
    labels = []

    if with_basic_stats:

        mean_x = np.mean(data, axis=1)
        std_x = np.std(data, axis=1, ddof=1)
        sum_x = np.sum(data, axis=1)

        data_transformed_list.append(mean_x)
        labels.append("Mean")
        data_transformed_list.append(std_x)
        labels.append("Std")
        data_transformed_list.append(sum_x)
        labels.append("Sum")

    if len(energy_grid) == 2:

        energies = energy_ref
        energy_bins_summary = np.append(energies[0], energies[1, -1])
        idx_low = np.searchsorted(energy_bins_summary, energy_grid.min())
        idx_high = np.searchsorted(energy_bins_summary, energy_grid.max())
        energy_bins_summary = energy_bins_summary[idx_low:idx_high + 1]

    else:
        energy_bins_summary = energy_grid

    counts = np.zeros((num_spectrum, len(energy_bins_summary),))
    energy_low_observation, energy_high_observation = energy_ref[:-1], energy_ref[1:]

    for i, (e_low_summary, e_high_summary) in enumerate(zip(energy_bins_summary[:-1], energy_bins_summary[1:])):
        counts_in_bin = np.sum(data[:, (energy_low_observation >= e_low_summary) & (energy_high_observation <= e_high_summary)], axis=1)
        counts[:, i] += counts_in_bin

        if with_sum:

            data_transformed_list.append(counts_in_bin)
            labels.append(f"Sums in band {e_low_summary:.4f}-{e_high_summary:.4f}")

    epsilon = 1
    # Hardness ratios
    if with_ratio:
        hardness_ratios = counts[:, 1:] / (counts[:, :-1] + epsilon)

        for i, (e_low_1, e_high_1, e_low_2, e_high_2) in enumerate(
                zip(
                    energy_bins_summary[:-2],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[2:]
                )):

            data_transformed_list.append(hardness_ratios[:, i])
            labels.append(f"Hardness ratio [{e_low_2:.2f}-{e_high_2:.2f}]/[{e_low_1:.2f}-{e_high_1:.2f}]")

    # Differential ratios
    if with_diff:
        differential_ratios = (counts[:, :-1] - counts[:, 1:]) / (counts[:, :-1] + counts[:, 1:] + epsilon)

        for i, (e_low_1, e_high_1, e_low_2, e_high_2) in enumerate(
                zip(
                    energy_bins_summary[:-2],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[1:-1],
                    energy_bins_summary[2:]
                )):

            data_transformed_list.append(differential_ratios[:, i])
            labels.append(f"Differential ratio [{e_low_2:.2f}-{e_high_2:.2f}]/[{e_low_1:.2f}-{e_high_1:.2f}]")

    if with_energy_weighted:
        for i, (e_low_summary, e_high_summary) in enumerate(zip(energy_bins_summary[:-1], energy_bins_summary[1:])):
            idx = (energy_low_observation >= e_low_summary) & (energy_high_observation <= e_high_summary)
            average_counts = data[:, idx]

            if average_counts.sum() < len(average_counts):
                average_counts = np.ones_like(average_counts)

            average_energy = (energy_low_observation[idx] + energy_high_observation[idx])/2
            result = np.apply_along_axis(lambda x : np.average(average_energy, weights=x/x.sum()), 1, average_counts)
            data_transformed_list.append(result)
            labels.append(f"Weighted energy in {e_low_summary:.4f}-{e_high_summary:.4f}")

    data_transformed = np.column_stack(data_transformed_list)

    return data_transformed, labels