import os
import time

import click
import dill as pickle
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from chainconsumer import ChainConsumer , PlotConfig , Chain , Truth
from jaxspec.data import ObsConfiguration
from matplotlib import pyplot as plt
from sbi import utils
from sbi.inference import SNPE
from sbi.utils import RestrictionEstimator , RestrictedPrior , get_density_thresholder
from tabulate import tabulate
from sixsa_utils import summary_statistics_func
from sixsa_utils import generate_function_for_cmin_cmax_restrictor , compute_x_sim , compute_cstat , \
    print_best_fit_parameters , plot_theta_in_theta_out , generate_function_for_cstat_restrictor , print_message


class sisxa_run :
    "This is where I define all the parameters used for running sixsa."
    creator = "Didier Barret"
    date = "2024-01-19"

    def __init__( self , yml_file ) :
        print("Initializing the SIXSA class")
        self.yml_file = yml_file
        with open(yml_file , 'r') as config_file :
            self.config = yaml.safe_load(config_file)

        # Create a list of tuples containing variable name and value pairs
        table_data = [(key , value) for key , value in  self.config.items( )]

        # Create a frame around the table and print it
        print(tabulate(table_data , headers = ["Variable" , "Value"] , tablefmt = "fancy_grid"))

        self.path_pha =  self.config['path_pha']
        self.reference_pha =  self.config['reference_pha']
        self.energy_min =  self.config['energy_range'][0]
        self.energy_max =  self.config['energy_range'][1]
        self.jaxspec_model_expression =  self.config['jaxspec_model_expression']
        self.parameter_lower_bounds =  self.config['parameter_lower_bounds']
        self.parameter_upper_bounds =  self.config['parameter_upper_bounds']
        self.parameter_prior_types =  self.config['parameter_prior_types']
        self.parameter_states =  self.config['parameter_states']
        self.parameter_names_for_plots =  self.config['parameter_names_for_plots']
        self.restricted_prior_type = self.config['restricted_prior_type']
        self.number_of_simulations_for_restricted_prior =  self.config['number_of_simulations_for_restricted_prior']
        if self.restricted_prior_type=='cmin_cmax_restricted_prior' :
            self.number_of_rounds_for_restricted_prior = self.config['number_of_rounds_for_restricted_prior']
            self.c_min_for_restricted_prior =  self.config['c_min_for_restricted_prior']
            self.c_max_for_restricted_prior =  self.config['c_max_for_restricted_prior']
            self.fraction_of_valid_simulations_to_stop_restricted_prior =  self.config[
                'fraction_of_valid_simulations_to_stop_restricted_prior']
        elif self.restricted_prior_type=='cstat_restricted_prior' :
            self.number_of_rounds_for_restricted_prior = self.config['number_of_rounds_for_restricted_prior']
            self.good_fraction_for_cstat_restricted_prior=self.config['good_fraction_for_cstat_restricted_prior']
        self.restricted_prior = None

        self.type_of_inference=self.config["type_of_inference"]
        if self.type_of_inference=='single round inference' :
            self.number_of_simulations_for_train_set=self.config["number_of_simulations_for_train_set"]
            self.number_of_simulations_for_test_set =  self.config['number_of_simulations_for_test_set']
        elif self.type_of_inference=='multiple round inference' :
            self.number_of_simulations_for_train_set =  self.config['number_of_simulations_for_train_set']
            self.number_of_rounds_for_multiple_inference=self.config["number_of_rounds_for_multiple_inference"]
        else :
            print_message("Invalid type_of_inference: can either be single round inference or multiple round inference ")

        self.number_of_posterior_samples =  self.config['number_of_posterior_samples']

        self.path_outputs =  self.config["path_outputs"]
        if not os.path.exists(self.path_outputs) :
            os.makedirs(self.path_outputs)
            print(f"Directory '{self.path_outputs}' created.")
        else :
            print(f"Directory '{self.path_outputs}' already exists.")

        self.root_output_files = os.path.basename(self.yml_file).replace(".yml" , "_")
        self.x_obs_reference=None
        self.use_summary=self.config.get("use_summary_statistics", False)

    def read_data_and_init_global_prior( self ):
        print("Read the PHA and initialize the global prior")
        self.pha_filename = self.path_pha + self.reference_pha
        print(self.pha_filename , self.energy_min , self.energy_max)

        # Translate to lower case for all parameters

        parameter_prior_types = list(map(str.lower , self.parameter_prior_types))
        parameter_states = list(map(str.lower , self.parameter_states))

        # Apply log10 transformation conditionally
        parameter_lower_bounds_transformed = np.where(
            np.array(parameter_prior_types) == "loguniform" ,
            np.log10(self.parameter_lower_bounds) ,
            self.parameter_lower_bounds
        )

        parameter_upper_bounds_transformed = np.where(
            np.array(parameter_prior_types) == "loguniform" ,
            np.log10(self.parameter_upper_bounds) ,
            self.parameter_upper_bounds
        )

        # Filter free parameters
        free_indices = [i for i , state in enumerate(parameter_states) if state == "free"]
        self.free_parameter_lower_bounds_transformed = parameter_lower_bounds_transformed[free_indices]
        self.free_parameter_upper_bounds_transformed = parameter_upper_bounds_transformed[free_indices]
        self.free_parameter_prior_types = [parameter_prior_types[i] for i in free_indices]
        self.free_parameter_names_for_plots = [self.parameter_names_for_plots[i] for i in free_indices]
        # Modify all elements in the second array based on the condition
        self.free_parameter_names_for_plots_transformed = [f"Log({n})" if pt.lower( ) == "loguniform" else n
                                                      for pt , n in
                                                      zip(parameter_prior_types , self.free_parameter_names_for_plots)]

        # Convert arrays to strings
        self.free_parameter_lower_bounds_transformed_str = ', '.join(map(str , self.free_parameter_lower_bounds_transformed))
        self.free_parameter_upper_bounds_transformed_str = ', '.join(map(str , self.free_parameter_upper_bounds_transformed))
        self.free_parameter_prior_types_str = ', '.join(map(str , self.free_parameter_prior_types))
        self.free_parameter_names_for_plots_str = ', '.join(map(str , self.free_parameter_names_for_plots))
        self.free_parameter_names_for_plots_transformed_str = ', '.join(
            map(str , self.free_parameter_names_for_plots_transformed))

        # Create a new table for the additional variables
        table_data_free_parameters = [
            ("Free Parameter Lower Bounds (Transformed)" , self.free_parameter_lower_bounds_transformed_str) ,
            ("Free Parameter Upper Bounds (Transformed)" , self.free_parameter_upper_bounds_transformed_str) ,
            ("Free Parameter Prior Types" , self.free_parameter_prior_types_str) ,
            ("Free Parameter Names for Plots" , self.free_parameter_names_for_plots_str) ,
            ("Free Parameter Names for Plots (Transformed)" , self.free_parameter_names_for_plots_transformed_str) ,
        ]

        # Create a frame around the new table and print it
        print(tabulate(table_data_free_parameters , headers = ["Variable" , "Value"] , tablefmt = "fancy_grid"))

        # Read the observed spectrum
        obs = ObsConfiguration.from_pha_file(self.pha_filename , low_energy = self.energy_min , high_energy = self.energy_max)
        self.e_min_folded = obs.e_min_folded
        self.e_max_folded = obs.e_max_folded

        num_bins = len(obs.folded_counts)
        total_counts = np.sum(obs.folded_counts)
        print(f"Number of bins {num_bins} - Exposure time {obs.exposure:.1f}s - Number of counts {total_counts:.1f}")
        self.x_obs_reference = np.array(obs.folded_counts)

        if self.use_summary:

            counts = np.array(obs.folded_counts)
            energy_ref = np.hstack([obs.out_energies[0], obs.out_energies[1][-1]])
            energy_grid = np.linspace(energy_ref.min(), energy_ref.max(), 10)

            def summary_func(x):
                return summary_statistics_func(x, energy_grid=energy_grid, energy_ref=energy_ref)

            self.summary_func = summary_func
            self.x_obs_summary = self.summary_func(counts)[0].squeeze()


        self.x_obs_reference_exposure_time = obs.exposure

        low_v = torch.as_tensor(self.free_parameter_lower_bounds_transformed)
        high_v = torch.as_tensor(self.free_parameter_upper_bounds_transformed)

        self.prior = utils.BoxUniform(low = low_v , high = high_v)

    def compute_restricted_prior( self ):

        if self.restricted_prior_type=="cmin_cmax_restricted_prior" :
            generate_restrictor_function_kwargs = {"cmin" : self.c_min_for_restricted_prior ,
                                                   "cmax" : self.c_max_for_restricted_prior}
            select_good_x = generate_function_for_cmin_cmax_restrictor(**generate_restrictor_function_kwargs)

            start_time_to_get_the_restricted_prior = time.perf_counter( )

            restriction_estimator = RestrictionEstimator(decision_criterion = select_good_x , prior = self.prior)
            restricted_prior_iterated = [self.prior]
            for r in range(self.number_of_rounds_for_restricted_prior) :
                print(f"Doing round {r + 1:d}")
                start_time = time.perf_counter( )
                theta_int = restricted_prior_iterated[-1].sample((self.number_of_simulations_for_restricted_prior ,))
                x_int = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , theta_int ,
                                      self.pha_filename ,
                                      self.energy_min ,
                                      self.energy_max ,
                                      self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = False ,
                                      verbose = False)

                end_time = time.perf_counter( )
                print(f'It took {end_time - start_time: 0.2f} second(s) to complete '
                      f'{self.number_of_simulations_for_restricted_prior:d} simulations at round {r + 1:d}')
                matching_x_int = [sub_array for sub_array in x_int.numpy( ) if
                                  self.c_min_for_restricted_prior <= sum(sub_array) <= self.c_max_for_restricted_prior]
                fraction_matching = len(matching_x_int) / len(x_int)

                print(
                    f"\nAt iteration {r + 1:d} - Fraction of x_int with sum between {self.c_min_for_restricted_prior} "
                    f"and {self.c_max_for_restricted_prior}: {fraction_matching}")
                if fraction_matching >= self.fraction_of_valid_simulations_to_stop_restricted_prior :
                    print("No need to continue - the restricted prior has converged")
                    break
                start_time = time.perf_counter( )
                restriction_estimator.append_simulations(theta_int , torch.as_tensor(x_int))

                if r < self.number_of_rounds_for_restricted_prior - 1 :
                    # The training is not needed in the last round because classifier will not be used anymore.
                    restriction_estimator.train( )

                restricted_prior_iterated.append(restriction_estimator.restrict_prior( ))

                end_time = time.perf_counter( )
                print(f'It took {end_time - start_time: 0.2f} second(s) to run the classifier on '
                      f'{self.number_of_simulations_for_restricted_prior:d} simulations at round {r + 1:d}')

            end_time = time.perf_counter( )
            self.duration_restricted_prior = end_time - start_time_to_get_the_restricted_prior
            print(f"The whole process took {self.duration_restricted_prior:.1f} seconds.")
            self.restricted_prior = restricted_prior_iterated[-1]

        elif self.restricted_prior_type == "cstat_restricted_prior":

            generate_restrictor_function_kwargs = {"x_obs" : np.array(self.x_obs_reference) ,
                                                   "good_fraction_in_percent" : self.good_fraction_for_cstat_restricted_prior}
            select_good_x = generate_function_for_cstat_restrictor(**generate_restrictor_function_kwargs)

            restriction_estimator = RestrictionEstimator(prior = self.prior , decision_criterion = select_good_x)
            restricted_prior_iterated = [self.prior]
            self.duration_restricted_prior = 0.
            start_time = time.perf_counter( )
            for r in range(self.number_of_rounds_for_restricted_prior) :

                theta_int = restricted_prior_iterated[-1].sample((self.number_of_simulations_for_restricted_prior ,))

                # This is the model that I need to compute the cstat, and therefore I put apply_stat=False

                x_int = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , theta_int ,
                                      self.pha_filename , self.energy_min ,
                                      self.energy_max ,
                                      self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = False ,
                                      verbose = False)

                restriction_estimator.append_simulations(theta_int , x_int)
                classifier = restriction_estimator.train( )
                restricted_prior_iterated.append(restriction_estimator.restrict_prior( ))
            end_time = time.perf_counter( )
            self.duration_restricted_prior  = end_time - start_time
            print(f"The whole process took {self.duration_restricted_prior:.1f} seconds.")
            self.restricted_prior = restricted_prior_iterated[-1]
        elif self.restricted_prior_type =="coarse_inference_restricted_prior" :
            start_time_to_get_the_restricted_prior = time.perf_counter( )
            theta_train = self.prior.sample((self.number_of_simulations_for_restricted_prior ,))
            print(f"Generating the simulations that will be used for the coarse inference")
            start_time = time.perf_counter( )
            x_train = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , theta_train ,
                                    self.pha_filename ,
                                    self.energy_min , self.energy_max ,
                                    self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                    verbose = False)

            end_time = time.perf_counter( )
            print(
                f"It took {end_time - start_time:0.2f} seconds to generate {self.number_of_simulations_for_restricted_prior :d} simulations")

            # Perform the coarse inference on a limited number of simulated spectra

            proposal = self.prior
            inference = SNPE(prior = self.prior)

            start_time = time.perf_counter( )
            density_estimator = inference.append_simulations(theta_train ,
                                                             torch.tensor(np.array(x_train).astype(np.float32)) ,
                                                             proposal = proposal).train( )
            posterior = inference.build_posterior(density_estimator)
            end_time = time.perf_counter( )
            self.duration_restricted_prior = end_time - start_time
            print(f"The whole process took { self.duration_restricted_prior:.1f} seconds.")

            self.restricted_prior = posterior.set_default_x(self.x_obs_reference)

    def plot_prior_and_restricted_priors( self ):
        # One can compare now the global prior and the restricted prior distributions

        theta_from_global_prior = self.prior.sample((10 * self.number_of_posterior_samples ,))
        df_theta_from_global_prior = pd.DataFrame(theta_from_global_prior ,
                                                  columns = self.free_parameter_names_for_plots_transformed)
        theta_from_restricted_prior = self.restricted_prior.sample((10*self.number_of_posterior_samples ,))
        df_theta_from_restricted_prior = pd.DataFrame(theta_from_restricted_prior ,
                                                      columns = self.free_parameter_names_for_plots_transformed)

        c = ChainConsumer( )
        c.set_plot_config(PlotConfig(usetex = True , serif = True , label_font_size = 18 , tick_font_size = 14))
        pdf_filename = self.path_outputs + self.root_output_files + "prior_and_restricted_prior_comparison.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
        c.add_chain(Chain(samples = df_theta_from_restricted_prior ,
                          name = f"Restricted prior" ,
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

    def generate_train_and_test_sets( self ):
        #
        # Now I am going to run the inference
        #
        self.theta_train = self.restricted_prior.sample((self.number_of_simulations_for_train_set ,))
        print(f"Generating the simulations that will be used for the inference")
        start_time = time.perf_counter( )
        self.x_train = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , self.theta_train ,
                                     self.pha_filename ,
                                     self.energy_min , self.energy_max ,
                                     self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                     verbose = False)
        if self.type_of_inference == "single round inference" :
            self.theta_test = self.restricted_prior.sample((self.number_of_simulations_for_test_set ,))
            self.x_test = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , self.theta_test , self.pha_filename ,
                                        self.energy_min , self.energy_max ,
                                        self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                        verbose = False)
        end_time = time.perf_counter( )
        print(f'It took {end_time - start_time: 0.2f} second(s) to complete the simulations to be used for the inference ')
        self.duration_generation_theta_x = end_time - start_time

    def plot_prior_predictive_check( self ):
        #
        # Let me perform a prior predictive check or restricted prior coverage check
        #
        pdf_filename = self.path_outputs + self.root_output_files + "prior_predictive_check.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
        fig , ax = plt.subplots(1 , 1)
        plt.step(0.5 * (self.e_min_folded + self.e_max_folded) , self.x_obs_reference , where = "mid" , color = "red" , linewidth = 2. , label = f"Observed spectrum ({np.int32(np.sum(self.x_obs_reference)):d} counts)")

        plt.fill_between(0.5 * (self.e_min_folded + self.e_max_folded),
            *np.percentile(self.x_train , [0. , 100] , axis = 0) ,
            color = "grey" ,
            alpha = 0.2 ,
            step = "mid" ,
            label = r"Restricted prior coverage")

        #
        # I want to have nice xticks values
        #

        self.logscale_values_low = np.logspace(np.log10(np.min(self.e_min_folded)) , np.log10(1.) , num = 5 , endpoint = False)
        self.logscale_values_high = np.logspace(np.log10(1.) , np.log10(np.max(self.e_max_folded)) , num = 6 , endpoint = True)
        self.logscale_values = np.concatenate((self.logscale_values_low , self.logscale_values_high))
        self.logscale_values_rounded = [round(val , 1) if val < 1 else int(val) for val in self.logscale_values]

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(self.logscale_values_rounded)
        ax.get_xaxis( ).set_major_formatter(matplotlib.ticker.ScalarFormatter( ))

        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        plt.legend(frameon = False)
        pdf.savefig( )
        matplotlib.pyplot.close( )
        pdf.close( )

    def run_single_round_inference_snpe( self ):

        proposal = self.prior
        inference = SNPE(prior = self.prior)

        if self.use_summary:
            raise NotImplementedError('Summary statistics are not implemented for single round inference.')

        start_time = time.perf_counter( )
        density_estimator = inference.append_simulations(self.theta_train ,
                                                         torch.tensor(np.array(self.x_train).astype(np.float32)) ,
                                                         proposal = proposal).train( )
        self.posterior = inference.build_posterior(density_estimator)
        end_time = time.perf_counter( )
        self.duration_inference = end_time - start_time
        print(f'It took {self.duration_inference: 0.2f} second(s) to run the inference')

    def run_multiple_round_inference_snpe( self ):

        multiple_round_inference_proposal = self.prior
        inference = SNPE(prior = self.prior)
        self.duration_generation_theta_x = 0.
        self.duration_inference = 0.

        for i_n_r in range(self.number_of_rounds_for_multiple_inference) :
            if i_n_r == 0 :
                # get the first sample from the restricted prior
                theta_train = self.restricted_prior.sample((self.number_of_simulations_for_train_set,))
                print(f"Generating the simulations that will be used for the inference at round {i_n_r + 1}")
                start_time = time.perf_counter( )
                x_train = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , theta_train ,
                                        self.pha_filename ,
                                        self.energy_min , self.energy_max ,
                                        self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                        verbose = False)
                end_time = time.perf_counter( )
                print(
                    f'It took {end_time - start_time: 0.2f} second(s) to complete {self.number_of_simulations_for_train_set :d} '
                    f'simulations to be used for the inference ')
                self.duration_generation_theta_x += end_time - start_time
            else :
                theta_train = multiple_round_inference_proposal.sample((self.number_of_simulations_for_train_set,))
                print(f"Generating the simulations that will be used for the inference at round {i_n_r + 1}")
                start_time = time.perf_counter( )
                x_train = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , theta_train ,
                                        self.pha_filename ,
                                        self.energy_min , self.energy_max ,
                                        self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = True ,
                                        verbose = False)
                end_time = time.perf_counter( )
                print(
                    f'It took {end_time - start_time: 0.2f} second(s) to complete {self.number_of_simulations_for_train_set :d} '
                    f'simulations to be used for the inference ')
                self.duration_generation_theta_x += end_time - start_time

            if self.use_summary:
                # Apply the summary statistics function to x_train
                start_time = time.perf_counter()
                x_train, _ = self.summary_func(np.asarray(x_train))
                x_train = torch.tensor(x_train.astype(np.float32))
                end_time = time.perf_counter()
                print(
                    f'It took {end_time - start_time: 0.2f} second(s) to compute the associated summary statistics')

            start_time = time.perf_counter( )
            density_estimator = inference.append_simulations(theta_train ,
                                                             torch.tensor(np.array(x_train).astype(np.float32)) ,
                                                             proposal = multiple_round_inference_proposal).train( )
            posterior_iterated = inference.build_posterior(density_estimator)
            end_time = time.perf_counter( )
            print(f'It took {end_time - start_time: 0.2f} second(s) to run the inference at round {i_n_r + 1:d}')
            self.duration_inference += end_time - start_time

            if self.use_summary:
                multiple_round_inference_proposal = posterior_iterated.set_default_x(self.x_obs_summary)
            else:
                multiple_round_inference_proposal = posterior_iterated.set_default_x(self.x_obs_reference)

#            accept_reject_fn = get_density_thresholder(multiple_round_inference_proposal , quantile = 1e-4)
#            multiple_round_inference_proposal = RestrictedPrior(self.prior , accept_reject_fn , sample_with = "rejection")
            if self.use_summary:
                posterior_samples = posterior_iterated.sample((self.number_of_posterior_samples ,) ,
                                                          x = torch.as_tensor(np.array(self.x_obs_summary)))
            else:
                posterior_samples = posterior_iterated.sample((self.number_of_posterior_samples ,) ,
                                                          x = torch.as_tensor(np.array(self.x_obs_reference)))

            median = np.median(posterior_samples , axis = 0)
            lower , upper = np.percentile(posterior_samples , (16 , 84) , axis = 0)
        #
            # Now computing the best fit model (setting apply_stat=False)
            #
            x_from_median = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , torch.tensor([median]) ,
                                          self.pha_filename ,
                                          self.energy_min , self.energy_max ,
                                          self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = False ,
                                          verbose = True)
            #
            # Now computing the cstat of the best fit and its deviation against the expected value
            # From Kaastra(2017) https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..51K/abstract
            #
            cstat_median_posterior_sample , cstat_dev_median_posterior_sample = compute_cstat(self.x_obs_reference ,
                                                                                              np.array(x_from_median) ,
                                                                                              verbose = True)
            print_message(f"\nAt iteration {i_n_r+1} - Best fit parameters\n")
            print_best_fit_parameters(self.x_obs_reference, self.free_parameter_names_for_plots ,
                                      self.free_parameter_prior_types , median , lower , upper ,
                                      cstat_median_posterior_sample , cstat_dev_median_posterior_sample)

        self.posterior=multiple_round_inference_proposal

    def plot_posterior_results_at_x_obs( self ):
        #
        # Now let us generate the poosterior samples at x_obs_reference
        #

        if self.use_summary:
            posterior_samples = self.posterior.sample((self.number_of_posterior_samples ,) ,
                                                  x = torch.as_tensor(np.array(self.x_obs_summary)))
        else:
            posterior_samples = self.posterior.sample((self.number_of_posterior_samples ,) ,
                                                  x = torch.as_tensor(np.array(self.x_obs_reference)))

        self.best_fit_parameters = np.median(posterior_samples , axis = 0)
        self.best_fit_parameters_lower_bounds , self.best_fit_parameters_upper_bounds = np.percentile(posterior_samples , (16 , 84) , axis = 0)

        # Find the maximum length of the labels for formatting
        max_label_length = max(len(label) for label in self.free_parameter_names_for_plots_transformed)
        max_widths = [max(len(label) , len(f"{med:.2f}") , len(f"{low:.2f}") , len(f"{up:.2f}")) for
                      label , med , low , up
                      in zip(self.free_parameter_names_for_plots_transformed , self.best_fit_parameters ,
                             self.best_fit_parameters_lower_bounds , self.best_fit_parameters_upper_bounds)]

        # Improved print statement with aligned values
        for label , med , low , up , width in zip(self.free_parameter_names_for_plots_transformed ,
                                                  self.best_fit_parameters ,
                                                  self.best_fit_parameters_lower_bounds ,
                                                  self.best_fit_parameters_upper_bounds ,
                                                  max_widths) :
            print(f"{label.ljust(max_label_length)}: Median={med:.2f}".ljust(width + 10) ,
                  f"Lower Percentile (16%)={low:.2f}".ljust(width + 10) , f"Upper Percentile (84%)={up:.2f}")
        #
        # Now computing the best fit model (setting apply_stat=False)
        #
        x_from_median = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , torch.tensor([
                                                                                                               self.best_fit_parameters]) ,
                                      self.pha_filename ,
                                      self.energy_min , self.energy_max ,
                                      self.free_parameter_prior_types , self.parameter_lower_bounds , apply_stat = False ,
                                      verbose = True)
        #
        # Now computing the cstat of the best fit and its deviation against the expected value
        # From Kaastra(2017) https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..51K/abstract
        #
        self.cstat_median_posterior_sample , self.cstat_dev_median_posterior_sample = compute_cstat(self.x_obs_reference ,
                                                                                                    np.array(x_from_median) ,
                                                                                                    verbose = True)

        print_best_fit_parameters(self.x_obs_reference, self.free_parameter_names_for_plots , self.free_parameter_prior_types ,
                                  self.best_fit_parameters , self.best_fit_parameters_lower_bounds ,
                                  self.best_fit_parameters_upper_bounds ,
                                  self.cstat_median_posterior_sample , self.cstat_dev_median_posterior_sample)

        #
        # Computing the residuals for the plot (assuming Gehrels errors)
        #
        gehrels_error_counts = (1. + (0.75 + np.array(self.x_obs_reference)) ** 0.5)
        best_fit_residuals = (np.array(self.x_obs_reference) - np.array(x_from_median)) / np.array(gehrels_error_counts)

        #
        # Now I compute x corresponding to the posterior samples at x_obs_reference
        #
        x_from_posterior_sample = compute_x_sim(self.jaxspec_model_expression , self.parameter_states , posterior_samples ,
                                                self.pha_filename ,
                                                self.energy_min , self.energy_max ,
                                                self.free_parameter_prior_types , self.parameter_lower_bounds ,
                                                apply_stat = True ,
                                                verbose = True)

        # create the dataframe for chain consumer

        df4cc = pd.DataFrame(posterior_samples , columns = self.free_parameter_names_for_plots_transformed)

        #
        # Do some plotting with chain consummer
        #
        c = ChainConsumer( )
        if self.type_of_inference =="single round inference"  :
            plot_title=f"SRI ({self.number_of_simulations_for_train_set :d} simulations)"
        elif self.type_of_inference =="multiple round inference":
            plot_title=f"MRI ({self.number_of_simulations_for_train_set :d} x {self.number_of_rounds_for_multiple_inference:d} simulations)"
        c.set_plot_config(PlotConfig(usetex = True , serif = True , label_font_size = 18 , tick_font_size = 14))
        pdf_filename = self.path_outputs + self.root_output_files + "posteriors_at_reference_spectrum.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
        c.add_chain(Chain(samples = df4cc , name = plot_title ,
                          color = "blue" , bar_shade = True))
        truth_sri = dict(zip(df4cc.columns.values.tolist( ) ,
                             np.array(df4cc.median( ))))
        c.add_truth(Truth(location = truth_sri , color = "blue"))
        fig = c.plotter.plot(figsize = (8 , 10))
        fig.align_ylabels( )
        fig.align_xlabels( )
        pdf.savefig( )
        matplotlib.pyplot.close( )
        pdf.close( )

        #
        # Plot the folded spectrum and the 68% percentile from the posterior samples at x_obs_reference
        #

        pdf_filename = self.path_outputs + self.root_output_files + "reference_spectrum_and_folded_model.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

        fig , ax = plt.subplots(2 , 1 , figsize = (8 , 10) , sharex = True , height_ratios = [0.8 , 0.2])
        plt.subplots_adjust(hspace = 0.0)

        # Plotting the data, best fit, and coverage
        ax[0].step(0.5 * (self.e_min_folded + self.e_max_folded) , self.x_obs_reference , where = "mid" , label = f"Observed spectrum ({np.int32(np.sum(self.x_obs_reference)):d} counts)" ,
                   color = "black")
        ax[0].step(0.5 * (self.e_min_folded + self.e_max_folded) , x_from_median.flatten( ) , where = "mid" ,
                   label = f"Best fit ({self.cstat_median_posterior_sample :0.1f}, {self.cstat_dev_median_posterior_sample :0.1f}$\sigma$)" ,
                   color = "blue")
        ax[0].fill_between(
            0.5 * (self.e_min_folded + self.e_max_folded) ,
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
        ax[0].set_title(plot_title)

        # Plotting residuals
        ax[1].step(0.5 * (self.e_min_folded + self.e_max_folded) , best_fit_residuals.flatten( ) , label = "Residuals" ,
                   color = "black")
        color = (0.15 , 0.25 , 0.45)
        ax[1].axhline(0 , color = color , ls = "--")
        ax[1].axhline(-3 , color = color , ls = ":")
        ax[1].axhline(3 , color = color , ls = ":")

        ax[1].set_yticks([-3 , 0 , 3] , labels = [-3 , 0 , 3])
        ax[1].set_yticks(range(-3 , 4) , minor = True)
        ax[1].set_ylabel("Residuals (" + r"$\sigma$)")

        ax[1].set_xticks(self.logscale_values_rounded , labels = self.logscale_values_rounded)
        ax[1].get_xaxis( ).set_major_formatter(matplotlib.ticker.ScalarFormatter( ))
        ax[1].set_xlabel("Energy (keV)")

        fig.align_ylabels( )
        fig.tight_layout( )

        pdf.savefig( )
        plt.close( )
        pdf.close( )

    def set_plot_layout( self ):
        print("Setting")
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

    def plot_posterior_results_at_x_test( self ):

        # Do some checks on the test sample : generate the posterior samples at all x_test, compute the median and compare
        # its median to the known theta_test - Generate a plot
        # If you had multiple observations, you could generate the posterior samples at your x_obs read from a list of pha files;
        #

        posterior_samples_at_x_test = []
        start_time = time.perf_counter( )
        for x_t in self.x_test :
            posterior_samples_at_x_test.append(
                self.posterior.sample((self.number_of_posterior_samples ,) , x = torch.tensor(np.array(x_t))))
        end_time = time.perf_counter( )
        self.duration_posterior_sample_generation_at_x_test = end_time - start_time
        pdf_filename = self.path_outputs + self.root_output_files + "parameter_in_out_at_test_set.pdf"
        plot_theta_in_theta_out(self.theta_test , posterior_samples_at_x_test ,
                                self.free_parameter_names_for_plots_transformed ,
                                pdf_filename)
    def print_run_summary( self  ):

        # Create a list of tuples containing variable name and value pairs
        table_data = [(key , value) for key , value in self.config.items( )]

        # Create a frame around the table and print it
        print(tabulate(table_data , headers = ["Variable" , "Value"] , tablefmt = "fancy_grid"))

        if self.type_of_inference == "single round inference" :
            table_data = [
                ("Restricted prior " , f"{self.duration_restricted_prior:.2f}") ,
                (
                f"Generation of {self.number_of_simulations_for_train_set + self.number_of_simulations_for_test_set:d} x_train and x_test" ,
                f"{self.duration_generation_theta_x:.2f}") ,
                ("Inference " , f"{self.duration_inference:.2f}") ,
                (f"Generation of {self.number_of_posterior_samples:d} posteriors at {self.number_of_simulations_for_test_set:d} x_test " ,
                 f"{self.duration_posterior_sample_generation_at_x_test:0.2f}")
            ]
        elif self.type_of_inference =="multiple round inference" :
            table_data = [
                ("Restricted prior " , f"{self.duration_restricted_prior:.2f}") ,
                (f"Generation of {self.number_of_simulations_for_train_set*self.number_of_rounds_for_multiple_inference} x_train" ,
                    f"{self.duration_generation_theta_x:.2f}") ,
                ("Inference " , f"{self.duration_inference:.2f}")
            ]

        # Create a frame around the table and print it
        print(tabulate(table_data , headers = ["Task Duration Summary" , "Seconds"] , tablefmt = "fancy_grid"))
        print_best_fit_parameters(self.x_obs_reference, self.free_parameter_names_for_plots , self.free_parameter_prior_types ,
                                  self.best_fit_parameters , self.best_fit_parameters_lower_bounds , self.best_fit_parameters_upper_bounds ,
                                  self.cstat_median_posterior_sample , self.cstat_dev_median_posterior_sample)

    def save_run_in_pickle_file( self ):
        self.pkl_filename=self.path_outputs + self.root_output_files + "run_results.pkl"

        if os.path.exists(self.pkl_filename) and click.confirm(
                f"{self.pkl_filename} exists. Do you still want to save the run results?" , default = False) :
            with open(self.pkl_filename , "wb") as handle :
                pickle.dump(self , handle , pickle.HIGHEST_PROTOCOL)
        else :
            with open(self.pkl_filename , "wb") as handle :
                pickle.dump(self , handle , pickle.HIGHEST_PROTOCOL)
