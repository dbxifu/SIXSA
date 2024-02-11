#
# This code shows how to perform single round inference for a spectrum contained in a reference pha file It reads the
# pha file, as to derive the parameters required for generating the faked spectra, the integration time, the response
# files, etc. You define a model simply, with the allowed range of parameters. For the inference to work, you need to
# define a prior with a given structure. The code performs sequentially these functions: 1) Read the data,
# 2) Define the "global" prior, 3) Define the restricted prior, 4) Generate the simulations, 5) Run the inference,
# 6) Generate the posterior samples. At each step, some plotting is performed to check that everything goes well. More
# information in Barret & Dupourqué (2024, A&A, in press, 10.48550/arXiv.2401.06061) To speed up the generation of
# simulated spectra, we use the jaxspec software under development (Dupourqué et al. 2024).
import matplotlib
import numpyro
import pandas as pd
import sbi
import scipy.stats
import yaml
from chainconsumer import ChainConsumer , PlotConfig , Truth , Chain
from matplotlib import pyplot as plt
from sbi import utils
from sbi.inference import SNPE
from sbi.utils import RestrictionEstimator

import time
import jax
import numpy as np
import torch
from sbi.utils.user_input_checks_utils import ScipyPytorchWrapper
from scipy.stats import loguniform
from tabulate import tabulate
from torch.distributions import ExpTransform , TransformedDistribution , Exponential , Uniform

from utils import compute_cstat , plot_theta_in_theta_out

numpyro.set_platform("cpu")
numpyro.set_host_device_count(6)
numpyro.enable_x64( )

import os
import warnings

from jaxspec.data import FoldingModel
from jaxspec.data.util import fakeit_for_multiple_parameters
from jaxspec.fit import BayesianModel
from jaxspec.model.abc import SpectralModel

import torch.distributions as dist

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

warnings.filterwarnings("ignore")

if __name__ == '__main__' :

    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.1
    plt.rcParams['xtick.direction'] = "in"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams["figure.figsize"] = (8 , 8)
    plt.rcParams["font.size"] = 18
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    path_yml_file="SIXA_YML_INPUT_FILES/"
    with open(path_yml_file+'sri_config.yml' , 'r') as config_file :
        config = yaml.safe_load(config_file)

    path_pha = config['path_pha']
    filter_pha = config['filter_pha']
    energy_min = config['energy_range'][0]
    energy_max = config['energy_range'][1]
    jaxspec_model_expression = config['jaxspec_model_expression']
    parameter_lower_bounds = config['parameter_lower_bounds']
    parameter_upper_bounds = config['parameter_upper_bounds']
    parameter_prior_types = config['parameter_prior_types']
    parameter_states = config['parameter_states']
    parameter_names_for_plots = config['parameter_names_for_plots']
    num_rounds_for_cmin_cmax_restrictor = config['num_rounds_for_cmin_cmax_restrictor']
    num_sim_for_cmin_cmax_restrictor = config['num_sim_for_cmin_cmax_restrictor']
    cmin_for_cmin_cmax_restrictor = config['cmin_for_cmin_cmax_restrictor']
    cmax_for_cmin_cmax_restrictor = config['cmax_for_cmin_cmax_restrictor']
    fraction_of_valid_simulations_to_stop_restricted_prior=config['fraction_of_valid_simulations_to_stop_restricted_prior']
    number_of_simulations_for_train_set = config['number_of_simulations_for_train_set']
    number_of_simulations_for_test_set = config['number_of_simulations_for_test_set']
    n_posterior_samples = config['n_posterior_samples']
    root_output_pdf_filename = config['root_output_pdf_filename']
    path_pdf_files = config["path_pdf_files"]
    if not os.path.exists(path_pdf_files):
        os.makedirs(path_pdf_files)
        print(f"Directory '{path_pdf_files}' created.")
    else:
        print(f"Directory '{path_pdf_files}' already exists.")


    # Create a list of tuples containing variable name and value pairs
    table_data = [(key , value) for key , value in config.items( )]

    # Create a frame around the table and print it
    print(tabulate(table_data , headers = ["Variable" , "Value"] , tablefmt = "fancy_grid"))

    pha_filename = path_pha + filter_pha
    print(pha_filename , energy_min , energy_max)

    # Translate to lower case for all parameters

    parameter_prior_types = list(map(str.lower , parameter_prior_types))
    parameter_states = list(map(str.lower , parameter_states))

    # Apply log10 transformation conditionally
    parameter_lower_bounds_transformed = np.where(
        np.array(parameter_prior_types) == "loguniform" ,
        np.log10(parameter_lower_bounds) ,
        parameter_lower_bounds
    )

    parameter_upper_bounds_transformed = np.where(
        np.array(parameter_prior_types) == "loguniform" ,
        np.log10(parameter_upper_bounds) ,
        parameter_upper_bounds
    )

    # Filter free parameters
    free_indices = [i for i , state in enumerate(parameter_states) if state == "free"]
    free_parameter_lower_bounds_transformed = parameter_lower_bounds_transformed[free_indices]
    free_parameter_upper_bounds_transformed = parameter_upper_bounds_transformed[free_indices]
    free_parameter_prior_types = [parameter_prior_types[i] for i in free_indices]
    free_parameter_names_for_plots = [parameter_names_for_plots[i] for i in free_indices]
    # Modify all elements in the second array based on the condition
    free_parameter_names_for_plots_transformed = [f"Log({n})" if pt.lower( ) == "loguniform" else n
                                                  for pt , n in zip(parameter_prior_types , free_parameter_names_for_plots)]

    # Convert arrays to strings
    free_parameter_lower_bounds_transformed_str = ', '.join(map(str , free_parameter_lower_bounds_transformed))
    free_parameter_upper_bounds_transformed_str = ', '.join(map(str , free_parameter_upper_bounds_transformed))
    free_parameter_prior_types_str = ', '.join(map(str , free_parameter_prior_types))
    free_parameter_names_for_plots_str = ', '.join(map(str , free_parameter_names_for_plots))
    free_parameter_names_for_plots_transformed_str = ', '.join(map(str , free_parameter_names_for_plots_transformed))

    # Create a new table for the additional variables
    table_data_free_parameters = [
        ("Free Parameter Lower Bounds (Transformed)" , free_parameter_lower_bounds_transformed_str) ,
        ("Free Parameter Upper Bounds (Transformed)" , free_parameter_upper_bounds_transformed_str) ,
        ("Free Parameter Prior Types" , free_parameter_prior_types_str) ,
        ("Free Parameter Names for Plots" , free_parameter_names_for_plots_str) ,
        ("Free Parameter Names for Plots (Transformed)" , free_parameter_names_for_plots_transformed_str) ,
    ]

    # Create a frame around the new table and print it
    print(tabulate(table_data_free_parameters , headers = ["Variable" , "Value"] , tablefmt = "fancy_grid"))

    # Read the observed spectrum
    obs = FoldingModel.from_pha_file(pha_filename , low_energy = energy_min , high_energy = energy_max)
    e_min_folded = obs.e_min_folded
    e_max_folded = obs.e_max_folded

    num_bins = len(obs.folded_counts)
    total_counts = np.sum(obs.folded_counts)
    print(f"Number of bins {num_bins} - Exposure time {obs.exposure:.1f}s - Number of counts {total_counts:.1f}")

    x_obs = np.array(obs.folded_counts)
    x_obs_exposure_time = obs.exposure

    low_v = torch.as_tensor(free_parameter_lower_bounds_transformed)
    high_v = torch.as_tensor(free_parameter_upper_bounds_transformed)

    prior = utils.BoxUniform(low = low_v , high = high_v)

    #
    # Define the restricted prior
    #

    print(f"{cmin_for_cmin_cmax_restrictor:.1f} {cmax_for_cmin_cmax_restrictor:.1f}")

    generate_restrictor_function_kwargs = {"cmin" : cmin_for_cmin_cmax_restrictor ,
                                           "cmax" : cmax_for_cmin_cmax_restrictor}
    select_good_x = generate_function_for_cmin_cmax_restrictor(**generate_restrictor_function_kwargs)

    start_time_to_get_the_restricted_prior = time.perf_counter( )

    from sbi.inference import SNPE
    from sbi.utils import RestrictionEstimator

    restriction_estimator = RestrictionEstimator(decision_criterion = select_good_x , prior = prior)
    cmin_cmax_restrictor = [prior]
    duration_cmin_cmax_restrictor = 0.
    for r in range(num_rounds_for_cmin_cmax_restrictor) :
        print(f"Doing round {r + 1:d}")
        start_time = time.perf_counter( )
        theta_int = cmin_cmax_restrictor[-1].sample((num_sim_for_cmin_cmax_restrictor ,))
        x_int = compute_x_sim(jaxspec_model_expression , parameter_states , theta_int , pha_filename , energy_min ,
                              energy_max ,
                              free_parameter_prior_types , parameter_lower_bounds , apply_stat = False , verbose = False)

        end_time = time.perf_counter( )
        print(f'It took {end_time - start_time: 0.2f} second(s) to complete '
              f'{num_sim_for_cmin_cmax_restrictor:d} simulations at round {r + 1:d}')
        matching_x_int = [sub_array for sub_array in x_int.numpy() if
                              cmin_for_cmin_cmax_restrictor <= sum(sub_array) <= cmax_for_cmin_cmax_restrictor]
        fraction_matching = len(matching_x_int) / len(x_int)

        print(f"Fraction of x_int with sum between {cmin_for_cmin_cmax_restrictor} and {cmax_for_cmin_cmax_restrictor}: {fraction_matching}")
        if fraction_matching >= fraction_of_valid_simulations_to_stop_restricted_prior :
            print("No need to continue - the restricted prior has converged")
            break
        start_time = time.perf_counter( )
        restriction_estimator.append_simulations(theta_int , torch.as_tensor(x_int))

        if r < num_rounds_for_cmin_cmax_restrictor - 1 :
            # The training is not needed in the last round because classifier will not be used anymore.
            restriction_estimator.train( )
        restricted_prior = restriction_estimator.restrict_prior( )

        cmin_cmax_restrictor.append(restricted_prior)
        end_time = time.perf_counter( )
        print(f'It took {end_time - start_time: 0.2f} second(s) to run the classifier on '
              f'{num_sim_for_cmin_cmax_restrictor:d} simulations at round {r + 1:d}')

    end_time = time.perf_counter( )
    duration_cmin_cmax_restrictor += end_time - start_time_to_get_the_restricted_prior
    print(f"The whole process took {duration_cmin_cmax_restrictor:.1f} seconds.")

    # One can compare now the global prior and the restricted prior distributions

    theta_from_global_prior = prior.sample((10 * n_posterior_samples ,))
    df_theta_from_global_prior = pd.DataFrame(theta_from_global_prior , columns = free_parameter_names_for_plots_transformed)
    theta_from_restricted_prior = cmin_cmax_restrictor[-1].sample((10 * n_posterior_samples ,))
    df_theta_from_restricted_prior = pd.DataFrame(theta_from_restricted_prior ,
                                                  columns = free_parameter_names_for_plots_transformed)

    c = ChainConsumer( )
    c.set_plot_config(PlotConfig(usetex = True , serif = True , label_font_size = 18 , tick_font_size = 14))
    pdf_filename = path_pdf_files+root_output_pdf_filename + "prior_and_restricted_prior_comparison.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
    c.add_chain(Chain(samples = df_theta_from_restricted_prior ,
                      name = f"Restricted prior ({num_sim_for_cmin_cmax_restrictor:d} x {num_rounds_for_cmin_cmax_restrictor:d})" ,
                      color = "green" , bar_shade = True))
    c.add_chain(Chain(samples = df_theta_from_global_prior ,
                      name = f"Global initial prior" ,
                      color = "blue" , bar_shade = True))

    fig = c.plotter.plot(figsize = (8 , 10))
    fig.align_ylabels( )
    fig.align_xlabels( )
    pdf.savefig( )
    matplotlib.pyplot.close( )
    pdf.close( )

    #
    # Now I am going to run the inference
    #
    theta_train = cmin_cmax_restrictor[-1].sample((number_of_simulations_for_train_set ,))
    theta_test = cmin_cmax_restrictor[-1].sample((number_of_simulations_for_test_set ,))

    print(f"Generating the simulations that will be used for the inference")
    start_time = time.perf_counter( )
    x_train = compute_x_sim(jaxspec_model_expression , parameter_states , theta_train , pha_filename ,
                            energy_min , energy_max ,
                            free_parameter_prior_types , parameter_lower_bounds , apply_stat = True , verbose = False)
    x_test = compute_x_sim(jaxspec_model_expression , parameter_states , theta_test , pha_filename ,
                           energy_min , energy_max ,
                           free_parameter_prior_types , parameter_lower_bounds , apply_stat = True , verbose = False)
    end_time = time.perf_counter( )
    print(
        f'It took {end_time - start_time: 0.2f} second(s) to complete {number_of_simulations_for_train_set + number_of_simulations_for_test_set:d} '
        f'simulations to be used for the inference ')
    duration_generation_theta_x = end_time - start_time

    #
    # Let me perform a prior predictive check or restricted prior coverage check
    #
    pdf_filename=path_pdf_files+root_output_pdf_filename + "Prior_predictive_check.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
    fig , ax = plt.subplots(1 , 1)
    plt.step(e_min_folded , x_obs , where = "post" , color = "red" , linewidth = 2. , label = "Observed spectrum")

    plt.fill_between(
        e_min_folded ,
        *np.percentile(x_train , [0. , 100] , axis = 0) ,
        color = "grey" ,
        alpha = 0.2 ,
        step = "post" ,
        label = r"Restricted prior coverage")

    #
    # I want to have nice xticks values
    #

    logscale_values_low = np.logspace(np.log10(np.min(e_min_folded)) , np.log10(1.) , num = 5 ,endpoint = False)
    logscale_values_high = np.logspace(np.log10(1.) , np.log10(np.max(e_max_folded)) , num = 6,endpoint = True)
    logscale_values = np.concatenate((logscale_values_low , logscale_values_high))
    logscale_values_rounded = [round(val , 1) if val < 1 else int(val) for val in logscale_values]
    print(logscale_values_rounded)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(logscale_values_rounded)
    ax.get_xaxis( ).set_major_formatter(matplotlib.ticker.ScalarFormatter( ))

    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.legend(frameon = False)
    pdf.savefig( )
    matplotlib.pyplot.close( )
    pdf.close( )

    # We are ready for the training - This is the heart of the code (straight from the sbi python package)

    proposal = prior
    inference = SNPE(prior = prior)

    start_time = time.perf_counter( )
    density_estimator = inference.append_simulations(theta_train ,
                                                     torch.tensor(np.array(x_train).astype(np.float32)) ,
                                                     proposal = proposal).train( )
    posterior = inference.build_posterior(density_estimator)
    end_time = time.perf_counter( )
    duration_inference = end_time - start_time
    print(f'It took {duration_inference: 0.2f} second(s) to run the inference')

    #
    # Now let us generate the poosterior samples at x_obs
    #
    posterior_samples = posterior.sample((n_posterior_samples ,) ,
                                         x = torch.as_tensor(np.array(x_obs)))

    median = np.median(posterior_samples , axis = 0)
    lower , upper = np.percentile(posterior_samples , (16 , 84) , axis = 0)

    # Find the maximum length of the labels for formatting
    max_label_length = max(len(label) for label in free_parameter_names_for_plots_transformed)
    max_widths = [max(len(label) , len(f"{med:.2f}") , len(f"{low:.2f}") , len(f"{up:.2f}")) for label , med , low , up
                  in zip(free_parameter_names_for_plots_transformed , median , lower , upper)]

    # Improved print statement with aligned values
    for label , med , low , up , width in zip(free_parameter_names_for_plots_transformed , median , lower , upper , max_widths) :
        print(f"{label.ljust(max_label_length)}: Median={med:.2f}".ljust(width + 10) ,
              f"Lower Percentile (16%)={low:.2f}".ljust(width + 10) , f"Upper Percentile (84%)={up:.2f}")
    input("This is your best fit - type enter to continue ")

    #
    # Now computing the best fit model (setting apply_stat=False)
    #
    x_from_median = compute_x_sim(jaxspec_model_expression , parameter_states , torch.tensor([median]) ,
                                  pha_filename ,
                                  energy_min , energy_max ,
                                  free_parameter_prior_types , parameter_lower_bounds , apply_stat = False ,
                                  verbose = True)
    #
    # Now computing the cstat of the best fit and its deviation against the expected value
    # From Kaastra(2017) https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..51K/abstract
    #
    cstat_median_posterior_sample , cstat_dev_median_posterior_sample = compute_cstat(x_obs , np.array(x_from_median) ,
                                                                                      verbose = True)

    #
    # Computing the residuals for the plot (assuming Gehrels errors)
    #
    gehrels_error_counts = (1. + (0.75 + np.array(x_obs)) ** 0.5)
    best_fit_residuals = (np.array(x_obs) - np.array(x_from_median)) / np.array(gehrels_error_counts)

    #
    # Now I compute x corresponding to the posterior samples at x_obs
    #
    x_from_posterior_sample = compute_x_sim(jaxspec_model_expression , parameter_states , posterior_samples ,
                                            pha_filename ,
                                            energy_min , energy_max ,
                                            free_parameter_prior_types , parameter_lower_bounds , apply_stat = True ,
                                            verbose = True)

    # create the dataframe for chain consumer

    df4cc = pd.DataFrame(posterior_samples , columns = free_parameter_names_for_plots_transformed)

    #
    # Do some plotting with chain consummer
    #
    c = ChainConsumer( )
    c.set_plot_config(PlotConfig(usetex = True , serif = True , label_font_size = 18 , tick_font_size = 14))
    pdf_filename = path_pdf_files+root_output_pdf_filename + "posteriors_at_reference_spectrum.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
    c.add_chain(Chain(samples = df4cc , name = f"Single Round Inference {number_of_simulations_for_train_set:d})" ,
                      color = "blue" , bar_shade = True))
    truth_sri = dict(zip(df4cc.columns.values.tolist( ) ,
                         np.array(df4cc.median( ))))
    print("truth_sri" , truth_sri)
    c.add_truth(Truth(location = truth_sri , color = "blue"))
    fig = c.plotter.plot(figsize = (8 , 10))
    fig.align_ylabels( )
    fig.align_xlabels( )
    pdf.savefig( )
    matplotlib.pyplot.close( )
    pdf.close( )

    #
    # Plot the folded spectrum and the 68% percentile from the posterior samples at x_obs
    #

    pdf_filename = path_pdf_files+root_output_pdf_filename + "reference_spectrum_and_folded_model.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

    fig , ax = plt.subplots(2 , 1 , figsize = (8 , 10) , sharex = True , height_ratios = [0.8 , 0.2])
    plt.subplots_adjust(hspace = 0.0)

    # Plotting the data, best fit, and coverage
    ax[0].step(0.5 * (e_min_folded + e_max_folded) , x_obs , where = "mid" , label = "Observed spectrum " ,
               color = "black")
    ax[0].step(0.5 * (e_min_folded + e_max_folded) , x_from_median.flatten( ) , where = "mid" ,
               label = f"Best fit ({cstat_median_posterior_sample:0.1f}, {cstat_dev_median_posterior_sample:0.1f}$\sigma$)" ,
               color = "blue")
    ax[0].fill_between(
        0.5 * (e_min_folded + e_max_folded) ,
        *np.percentile(x_from_posterior_sample , [16 , 84] , axis = 0) ,
        alpha = 0.3 ,
        color = "green" ,
        step = "mid" ,
        label = r"$1-\sigma$ coverage" ,
    )
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].set_ylabel("Counts")
    ax[0].legend(frameon = False)
    ax[0].set_title(f"Single round inference with a training sample of {number_of_simulations_for_train_set:d} spectra")

    # Plotting residuals
    ax[1].step(0.5 * (e_min_folded + e_max_folded) , best_fit_residuals.flatten( ) , label = "Residuals" ,
               color = "black")
    color = (0.15 , 0.25 , 0.45)
    ax[1].axhline(0 , color = color , ls = "--")
    ax[1].axhline(-3 , color = color , ls = ":")
    ax[1].axhline(3 , color = color , ls = ":")

    ax[1].set_yticks([-3 , 0 , 3] , labels = [-3 , 0 , 3])
    ax[1].set_yticks(range(-3 , 4) , minor = True)
    ax[1].set_ylabel("Residuals (" + r"$\sigma$)")

    ax[1].set_xticks(logscale_values_rounded , labels = logscale_values_rounded)
    ax[1].get_xaxis( ).set_major_formatter(matplotlib.ticker.ScalarFormatter( ))
    ax[1].set_xlabel("Energy (keV)")

    fig.align_ylabels( )
    fig.tight_layout( )

    pdf.savefig( )
    plt.close( )
    pdf.close( )

    #
    # Do some checks on the test sample : generate the posterior samples at all x_test, compute the median and compare
    # its median to the known theta_test - Generate a plot
    # If you had multiple observations, you could generate the posterior samples at those x_obs
    #

    posterior_samples_at_x_test = []
    start_time = time.perf_counter( )
    for x_t in x_test :
        posterior_samples_at_x_test.append(
            posterior.sample((n_posterior_samples ,) , x = torch.tensor(np.array(x_t))))
    end_time = time.perf_counter( )
    duration_posterior_sample_generation_at_x_test = end_time - start_time
    pdf_filename = path_pdf_files+root_output_pdf_filename + "parameter_in_out_at_test_set.pdf"
    plot_theta_in_theta_out(theta_test , posterior_samples_at_x_test , free_parameter_names_for_plots_transformed , pdf_filename)

    # Create a list of tuples containing variable name and value pairs
    table_data = [(key , value) for key , value in config.items( )]

    # Create a frame around the table and print it
    print(tabulate(table_data , headers = ["Variable" , "Value"] , tablefmt = "fancy_grid"))

    table_data = [
        ("Restrictor " , f"{duration_cmin_cmax_restrictor:.2f}") ,
        (f"Generation of {number_of_simulations_for_train_set:d} x_train" , f"{duration_generation_theta_x:.2f}") ,
        ("Inference " , f"{duration_inference:.2f}") ,
        (f"Generation of {n_posterior_samples:d} posteriors at {number_of_simulations_for_test_set:d} x_test ", f"{duration_posterior_sample_generation_at_x_test:0.2f}")
    ]

    # Create a frame around the table and print it
    print(tabulate(table_data , headers = ["Task Duration Summary" , "Seconds"] , tablefmt = "fancy_grid"))

    print("We are done with running this very simple example ! I hope you enjoyed it !\nNow you can customize if for your application !")
