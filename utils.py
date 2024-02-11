import numpy as np
import torch
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


def db_get_x_from_thetas( pha_file , energy_min , energy_max , jaxspec_model_str , parameter_state , parameter_prior ,
                          parameter_lower_bounds , thetas , apply_stat = True , verbose = True ) :
    print("in db_get_x_from_theta")

    jaxspec_model = SpectralModel.from_string(jaxspec_model_str)
    print("jaxspec_model.params=" , jaxspec_model.params)
    parameter_values = []
    index_theta = 0
    for i_param in range(jaxspec_model.n_parameters) :
        if parameter_state[i_param].lower( ) == "free" and parameter_prior[i_param].lower( ) == "uniform" :
            parameter_values.append([thetas[j][index_theta] for j in range(len(thetas))])
            index_theta += 1
            if verbose : print(
                f"{parameter_state[i_param].lower( )} Parameter #{i_param + 1} of {jaxspec_model.n_parameters} is {parameter_prior[i_param].lower( )} ")
        if parameter_state[i_param].lower( ) == "free" and parameter_prior[i_param].lower( ) == "loguniform" :
            parameter_values.append([10. ** thetas[j][index_theta] for j in range(len(thetas))])
            index_theta += 1
            if verbose : print(
                f"{parameter_state[i_param].lower( )} Parameter #{i_param + 1} of {jaxspec_model.n_parameters} is {parameter_prior[i_param].lower( )} ")
        if parameter_state[i_param].lower( ) == "frozen" :
            parameter_values.append([parameter_lower_bounds[i_param] for j in range(len(thetas))])
            if verbose : print(
                f"{parameter_state[i_param].lower( )} Parameter #{i_param + 1} of {jaxspec_model.n_parameters} is {parameter_prior[i_param].lower( )} ")

    params_to_set = jaxspec_model.params
    i_para = 0
    for l in list(params_to_set.keys( )) :
        for i_p in range(len(list(params_to_set[l].keys( )))) :
            upd_dict = {list(params_to_set[l].keys( ))[i_p] : np.array(parameter_values[i_para])}
            params_to_set[l].update(upd_dict)
            i_para += 1

    #   print( params_to_set )

    folding_model = FoldingModel.from_pha_file(pha_file , energy_min , energy_max)
    start_time = time.perf_counter( )
    x = jax.jit(lambda s : fakeit_for_multiple_parameters(folding_model , jaxspec_model , s , apply_stat = apply_stat))(
        params_to_set)
    end_time = time.perf_counter( )
    duration_time = end_time - start_time
    print(f"Run duration_time {duration_time:0.1f} seconds for {len(thetas)} samples")
    return x


def generate_function_for_cmin_cmax_restrictor( cmin = 2000. , cmax = 5000. ) :
    def get_good_x( x ) :
        good_x_array = []
        n_bad = 0
        for x_p in x :
            good_x = False
            if cmin <= np.sum(x_p.numpy( )) <= cmax :
                good_x = True
            #                print(np.sum(x_p.numpy()),good_x)
            else :
                #                print(np.sum(x_p.numpy()),good_x)
                n_bad += 1
            good_x_array.append(good_x)
        print(
            f"{cmin:0.1f} {cmax:0.1f} number of simulations outside the range {n_bad:d} - "
            f"number of good simulation {len(good_x_array) - n_bad:d} - "
            f"good fraction = {100. * np.float32(1. - n_bad / len(good_x_array)):0.1f}")
        return torch.as_tensor(good_x_array)

    return get_good_x


def define_initial_prior_from_model_parameters( parameter_names , parameter_lower_bounds , parameter_upper_bounds ,
                                                parameter_priors , parameter_state , verbose = True ) :
    low_v = []
    high_v = []
    for i_p_p in range(len(parameter_priors)) :
        if parameter_priors[i_p_p].lower( ) == "uniform" and parameter_state[i_p_p].lower( ) == "free" :
            low_v.append(parameter_lower_bounds[i_p_p])
            high_v.append(parameter_upper_bounds[i_p_p])
            if verbose : print(
                f"{parameter_state[i_p_p].lower( )} {parameter_names[i_p_p]} Parameter #{i_p_p + 1} of {len(parameter_priors)} is {parameter_priors[i_p_p].lower( )} between {low_v[-1]:0.3f}/{parameter_lower_bounds[i_p_p]:0.3f} and {high_v[-1]:0.3f}/{parameter_upper_bounds[i_p_p]:0.3f}")
        elif parameter_priors[i_p_p].lower( ) == "loguniform" and parameter_state[i_p_p].lower( ) == "free" :
            low_v.append(np.log10(parameter_lower_bounds[i_p_p]))
            high_v.append(np.log10(parameter_upper_bounds[i_p_p]))
            if verbose : print(
                f"{parameter_state[i_p_p].lower( )} {parameter_names[i_p_p]} Parameter #{i_p_p + 1} of {len(parameter_priors)} is {parameter_priors[i_p_p].lower( )} between {low_v[-1]:0.3f}/{parameter_lower_bounds[i_p_p]:0.3f} and {high_v[-1]:0.3f}/{parameter_upper_bounds[i_p_p]:0.3f}")
        elif parameter_state[i_p_p].lower( ) == "frozen" :
            if verbose : print(
                f"{parameter_state[i_p_p].lower( )} {parameter_names[i_p_p]} Parameter #{i_p_p + 1} of {len(parameter_priors)} is set to {parameter_lower_bounds[i_p_p]:0.3f}")

    initial_prior = utils.BoxUniform(low = torch.as_tensor(low_v) , high = torch.as_tensor(high_v))

    return initial_prior


import haiku as hk
import jax.numpy as jnp
import numpyro.distributions as dist


def data_decomposition_before_training( x_train_in , x_test_in , x_obs_in , threshold_pca_decomposition = 0.995 ,
                                        number_of_components = -1 ) :
    scaler = StandardScaler( )
    all_x_combined = torch.cat(
        (torch.tensor(np.array(x_train_in).astype(np.float32)) , torch.tensor(np.array(x_test_in).astype(np.float32))) ,
        0)
    all_x_combined = torch.cat((all_x_combined , torch.tensor(np.array(x_obs_in).astype(np.float32))) , 0)
    scaler.fit(all_x_combined)
    all_x_combined_s = torch.tensor(scaler.transform(all_x_combined).astype(np.float32))
    print("Starting the PCA decomposition")
    pca = PCA(n_components = len(all_x_combined_s[0]) , svd_solver = 'auto')
    pca.fit(all_x_combined_s)
    Principal_components = pca.fit_transform(all_x_combined_s)
    PC_values = np.arange(pca.n_components_) + 1
    print("Proportion of Variance Explained : " , pca.explained_variance_ratio_)
    out_sum = np.cumsum(pca.explained_variance_ratio_)

    print(
        f"Order decomposition to reach {threshold_pca_decomposition * 100.:0.5f}% == {np.min(np.where(out_sum > threshold_pca_decomposition)) + 1:d}")
    n_components_required = np.min(np.where(out_sum > threshold_pca_decomposition)) + 1
    print(
        f"n_components_required: {n_components_required:d} - number_of_components as an input {number_of_components:d}")
    if number_of_components > 0 : n_components_required = number_of_components
    pca = PCA(n_components = n_components_required , svd_solver = 'auto')
    pca.fit(all_x_combined_s)
    all_x_combined_s_p = torch.tensor(pca.fit_transform(all_x_combined_s).astype(np.float32))
    print("np.shape(all_x_combined_s_p)=" , np.shape(all_x_combined_s_p))

    x_train_out = torch.tensor(np.array(all_x_combined_s_p[0 :len(x_train_in)]).astype(np.float32))
    x_test_out = torch.tensor(
        np.array(all_x_combined_s_p[len(x_train_in) :len(x_train_in) + len(x_test_in)]).astype(np.float32))
    x_obs_out = torch.tensor(np.array(all_x_combined_s_p[len(x_train_in) + len(x_test_in) :]).astype(np.float32))

    return [x_train_out , x_test_out , x_obs_out , n_components_required]


def build_prior_for_jaxspec_fitting(
        jaxspec_model_str ,
        parameter_lower_bounds ,
        parameter_upper_bounds ,
        parameter_prior_types ,
        parameter_states
) :
    # Courtesy of Simon 23/01/2024

    jaxspec_model = SpectralModel.from_string(jaxspec_model_str)
    params_to_set = jaxspec_model.params
    prior_parameters = hk.data_structures.to_haiku_dict(jaxspec_model.params)
    index = 0

    for component in params_to_set.keys( ) :
        for parameter in params_to_set[component].keys( ) :
            if parameter_states[index] == "free" :

                match parameter_prior_types[index] :
                    case "uniform" :
                        dist_to_set = dist.Uniform
                    case "loguniform" :
                        dist_to_set = dist.LogUniform
                    case _ :
                        raise TypeError('Supported distributions are Uniform and LogUniform')

                prior_parameters[component][parameter] = dist_to_set(parameter_lower_bounds[index] ,
                                                                     parameter_upper_bounds[index])

            else :

                prior_parameters[component][parameter] = jnp.asarray(parameter_lower_bounds[index])

            index += 1
    print(prior_parameters)
    return prior_parameters


def get_parameter_names_from_jaxspec_model( jaxspec_model_str ) :
    jaxspec_model = SpectralModel.from_string(jaxspec_model_str)
    params_to_set = jaxspec_model.params
    parameter_names = []
    component_parameter_names = []
    for component in params_to_set.keys( ) :
        for parameter in params_to_set[component].keys( ) :
            print(component , parameter)
            parameter_names.append(parameter)
            component_parameter_names.append(component + "_" + parameter)
            print(parameter_names[-1] , component_parameter_names[-1])
    return parameter_names , component_parameter_names


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


def plot_pha_with_samples( pha_file , energy_min , energy_max , samples , x_from_median_samples = [] ,
                           pdf_filename = "tmp.pdf" ) :
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
    fig , ax = plt.subplots(tight_layout = True)

    obs = FoldingModel.from_pha_file(pha_file , low_energy = energy_min , high_energy = energy_max)
    ax.step(np.mean(obs.out_energies , axis = 0) , obs.folded_counts , where = "mid" , label = r"Data" ,
            color = "black")
    if len(x_from_median_samples) > 0 : ax.step(np.mean(obs.out_energies , axis = 0) , x_from_median_samples ,
                                                where = "mid" , label = r"Sample median" , color = "blue" ,
                                                linewidth = 3)

    ax.fill_between(
        np.mean(obs.out_energies , axis = 0) ,
        *np.percentile(samples , [0. , 100] , axis = 0) ,
        color = "green" ,
        alpha = 0.3 ,
        step = "post" ,
        label = "Coverage"
    )

    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    _ = plt.loglog( )
    plt.legend( )
    pdf.savefig( )
    matplotlib.pyplot.close( )
    pdf.close( )


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

