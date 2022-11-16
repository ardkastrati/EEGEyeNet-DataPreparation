##################################################################
# Data preparation configurations
import time
import os
import numpy as np

preparation_config = dict()

# The task for which we want to prepare the data. Possible choices that are implemented so far are:
# 'LR_task' (dataset: 'antisaccade'):
# 'Direction_task' (dataset: 'dots' or 'processing_speed'):
# 'Position_task' (dataset: 'dots'):
# 'Segmentation_task' (dataset: 'antisaccade', 'dots', or 'processing_speed'):
# 'Yes_No_task' (dataset: 'wisc' and 'synchronized EEG'):
preparation_config['task']    = 'Yes_No_task'
preparation_config['dataset'] = 'WISC' #'antisaccade'

# We provide two types of preprocessing on the dataset (minimal preprocessing and maximal preprocessing). Choices are
# 'max'
# 'min'
preparation_config['preprocessing']      = 'min'  
preparation_config['preprocessing_path'] = 'synchronized_' + preparation_config['preprocessing']

# We provide also dataset where features are extracted
# (typically used for training with standard machine learning methods).
# The feature extraction that we have implemented is hilbert transformed data for phase and amplitude.
preparation_config['feature_extraction'] = False

# Maybe for later we can also include the bandpassed data on
# top of the feature extracted data (this is not implemented yet).
preparation_config['including_bandpass_data'] = False  # or True (for later)

#The directory of output file and the name
preparation_config['SAVE_PATH'] = '../data/prepared/'
preparation_config['output_name'] = preparation_config['task'] + '_with_' + preparation_config['dataset']
preparation_config['output_name'] = preparation_config['output_name'] + '_' + preparation_config['preprocessing_path']
preparation_config['output_name'] = preparation_config['output_name'] + ('_hilbert.npz' if preparation_config['feature_extraction'] else '.npz')

##################################################################################
# We prepare some helper variables to locate the correct datasets and the files that we need and to use them.
preparation_config['LOAD_ANTISACCADE_PATH'] = '../data/measured/antisaccade_task_data/' + preparation_config['preprocessing_path'] + '/'
preparation_config['ANTISACCADE_FILE_PATTERN']         = '[go]ip_..._AS_EEG.mat'
preparation_config['ANTISACCADE_HILBERT_FILE_PATTERN'] = '[go]ip_..._AS_EEG.mat'

preparation_config['LOAD_DOTS_PATH'] = '../data/measured/dots_data/' + preparation_config['preprocessing_path'] + '/'
preparation_config['DOTS_FILE_PATTERN']         = '(ep|EP).._DOTS._EEG.mat'
preparation_config['DOTS_HILBERT_FILE_PATTERN'] = '(ep|EP).._DOTS._EEG.mat'

preparation_config['LOAD_PROCESSING_SPEED_PATH'] = '../data/measured/processing_speed_data/' + preparation_config['preprocessing_path'] + '/'
preparation_config['PROCESSING_SPEED_FILE_PATTERN']         = '..._WI2_EEG.mat'
preparation_config['PROCESSING_SPEED_HILBERT_FILE_PATTERN'] = '..._WI2_EEG.mat'

preparation_config['LOAD_WISC_PATH']         = '.../data/measured/'
preparation_config['WISC_FILE_PATTERN']      = '..._W[12]_wisc_events.npz'


##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
# Internal information about each dataset (antisaccade, dots, processing_speeed)
preparation_config['saccade_trigger']  = ['L_saccade', 'R_saccade']
preparation_config['fixation_trigger'] = ['L_fixation', 'R_fixation']
preparation_config['blink_trigger']    = ['L_blink', 'R_blink']

# Anti-saccade dataset
preparation_config['antisaccade'] = dict()
preparation_config['antisaccade']['cue_trigger'] = ['10', '11']
preparation_config['antisaccade']['matlab_struct'] = 'EEG'

#Dots dataset
preparation_config['dots'] = dict()
preparation_config['dots']['cue_trigger'] = list(map(str, range(1, 28))) + list(map(str, range(101, 128)))
preparation_config['dots']['end_cue_trigger'] =['41']
preparation_config['dots']['matlab_struct'] = 'sEEG'
preparation_config['dots']['tar_pos'] = np.array([
        [400, 300], [650, 500], [400, 100], [100, 450], [700, 450], [100, 500],
        [200, 350], [300, 400], [100, 150], [150, 500], [150, 100], [700, 100],
        [300, 200], [100, 100], [700, 500], [500, 400], [600, 250], [650, 100],
        [400, 300], [200, 250], [400, 500], [700, 150], [500, 200], [100, 300],
        [700, 300], [600, 350], [400, 300]
    ])

# Processing speed dataset
preparation_config['processing_speed'] = dict()
preparation_config['processing_speed']['matlab_struct'] = 'sEEG'

#Maybe we should do logging here as well ...