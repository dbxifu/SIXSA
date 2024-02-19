#
# This code shows how to perform multiple round inference for a spectrum contained in a reference pha file It reads the
# pha file, as to derive the parameters required for generating the faked spectra, the integration time, the response
# files, etc. You define a model simply, with the allowed range of parameters. For the inference to work, you need to
# define a prior with a given structure. The code performs sequentially these functions: 1) Read the data,
# 2) Define the "global" prior, 3) Define the restricted prior, 4) Generate the simulations, 5) Run the inference,
# 6) Generate the posterior samples. At each step, some plotting is performed to check that everything goes well. More
# information in Barret & Dupourqué (2024, A&A, in press, 10.48550/arXiv.2401.06061) To speed up the generation of
# simulated spectra, we use the jaxspec software under development (Dupourqué et al. 2024).
import glob
import os
import warnings

import numpy as np

from sixsa_utils import welcome_message , goodbye_message , \
    robust_selection_from_menu , extract_and_print_imports , print_message

warnings.filterwarnings("ignore")
from sixsa_class import sisxa_run

if __name__ == '__main__' :
    extract_and_print_imports()
    welcome_message()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    user_selection=robust_selection_from_menu("Select the type of inference to run",["Run single round inference","Run multiple round inference"],
                                              return_index = False)
    path_yml_file= "SISXA_YML_INPUT_FILES/"
    if user_selection == "Run single round inference" :
        yml_files=np.sort(glob.glob(path_yml_file+"*_sri_config_*.yml"))
    elif user_selection == "Run multiple round inference" :
        yml_files=np.sort(glob.glob(path_yml_file+"*_mri_config_*.yml"))

    index_yml_file_selected=robust_selection_from_menu("Select the yml file to use as input",yml_files)
    demo_run=sisxa_run(yml_files[index_yml_file_selected-1])
    print_message("\nReading data\n")
    demo_run.read_data_and_init_global_prior()
    print_message("\nComputing the restricted prior\n")
    demo_run.compute_restricted_prior()
    demo_run.set_plot_layout()
    print_message("\nPlotting the restricted prior\n")
    demo_run.plot_prior_and_restricted_priors()
    demo_run.generate_train_and_test_sets()
    demo_run.plot_prior_predictive_check()
    print_message("\nRunning the inference\n")
    if user_selection == "Run single round inference" :
        demo_run.run_single_round_inference_snpe( )
    elif user_selection == "Run multiple round inference" :
        demo_run.run_multiple_round_inference_snpe( )
    print_message("\nPlotting the posteriors at x_obs\n")
    demo_run.plot_posterior_results_at_x_obs()
    if user_selection == "Run single round inference" :
        print_message("\nPlotting the posteriors at x_test\n")
        demo_run.plot_posterior_results_at_x_test( )

    demo_run.print_run_summary()
    print_message("\nSaving the run\n")
    demo_run.save_run_in_pickle_file()
    goodbye_message()