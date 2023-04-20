#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
import joblib
import time
import zlib
import antropy as ant
import math
from itertools import permutations
from mne.utils import logger, _time_mask
from mne_connectivity import spectral_connectivity_epochs
from sklearn.gaussian_process.kernels import RBF
from distfit import distfit

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Load data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # Extract features.
        current_features = get_features(patient_metadata, recording_metadata, recording_data)
        features.append(current_features)

        # Extract labels.
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)
    
    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')
    
    # initialize models,preprocessing and imputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    scaler = StandardScaler()
    from xgboost import XGBRegressor,XGBClassifier
    from lightgbm import LGBMRegressor,LGBMClassifier
    from sklearn.linear_model import BayesianRidge,TweedieRegressor,OrthogonalMatchingPursuit,LogisticRegression,LassoCV
    from sklearn.svm import SVR,NuSVR,LinearSVC,SVC
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import VotingRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor,\
        RandomForestRegressor,StackingRegressor,RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier,\
            AdaBoostClassifier,BaggingClassifier,VotingClassifier,StackingClassifier,StackingRegressor
    from sklearn.neural_network import MLPRegressor,MLPClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.impute import KNNImputer
    from sklearn.utils import all_estimators
    from sklearn.base import ClassifierMixin,RegressorMixin
    from sklearn.feature_selection import mutual_info_classif, SequentialFeatureSelector, SelectPercentile, f_regression,f_classif
    from scipy.stats import zscore,iqr

    
    # what models made the cut
    pipe_grid = {'AdaBoostRegressor':{"Scaler":None,"Features":"all"},\
                  'BayesianRidge':{"Scaler":"normal","Features":"all",\
                                   "feature selection":\
                                       SelectPercentile(score_func = f_regression, percentile = 99)},\
                  'GradientBoostingRegressor':{"Scaler":None,"Features":"all",\
                                               "feature selection": SelectPercentile(score_func = f_regression, percentile = 70)},\
                  'HistGradientBoostingRegressor':{"Scaler":None,"Features":"all",\
                                                   "feature selection": SelectPercentile(score_func = f_regression, percentile = 10)},\
                  'KNeighborsRegressor':{"Scaler":"normal","Features":"all"},\
                  'MLPRegressor':{"Scaler":None,"Features":"all"},\
                  'OrthogonalMatchingPursuit':{"Scaler":"normal","Features":"all"},\
                  'NuSVR':{"Scaler":"normal","Features":"all"},\
                  'RandomForestRegressor':{"Scaler":None,"Features":"all",\
                                           "feature selection": SelectPercentile(score_func = f_regression, percentile = 1)},\
                  'TweedieRegressor':{"Scaler":"normal","Features":"all"},\
                  'SVR':{"Scaler":"normal","Features":"all"},\
                  'XGBRegressor':{"Scaler":None,"Features":"all",\
                                  "feature selection":SelectPercentile(score_func = f_regression, percentile = 50)},\
                  'LGBMRegressor':{"Scaler":None,"Features":"all",\
                      "feature selection":SelectPercentile(score_func = f_regression, percentile = 2)}}
    
    param_grid = {'AdaBoostRegressor':{"n_estimators":10,"learning_rate":0.01},\
                  'BayesianRidge':{"alpha_1":1e-3,"alpha_2":1e-3,"lambda_1":1e-3,"lambda_2":1e-3},\
                  'GradientBoostingRegressor':{'n_estimators':1000,'learning_rate':0.04,\
                                                    "min_samples_split":0.1,"min_samples_leaf":0.05,\
                                                    'min_weight_fraction_leaf':0.1,'max_features':"log2",\
                                                        "ccp_alpha":0.0,'subsample':0.90},\
                  'HistGradientBoostingRegressor':{'learning_rate':0.01,'max_depth':5,"max_leaf_nodes":None,\
                                                    'min_samples_leaf':5,'l2_regularization':0.01,"max_iter":100,\
                                                        "interaction_cst":"no_interactions"},\
                  'KNeighborsRegressor':{"n_neighbors":20,'weights':'distance','algorithm':'auto','p':1},\
                  'MLPRegressor':{"hidden_layer_sizes":(1000,100),"activation":"relu","alpha":0.1,\
                                   "learning_rate":"adaptive","momentum":0.95,"early_stopping":False},\
                  'OrthogonalMatchingPursuit':{"n_nonzero_coefs":20},\
                  'NuSVR':{"nu":0.9,"C":10,"gamma":"auto"},\
                  'RandomForestRegressor':{"max_depth":None,"n_estimators":1000,"min_samples_split":0.1,"min_samples_leaf":0.05,\
                                                'min_weight_fraction_leaf':0.0,'max_leaf_nodes':None,'max_features':"log2",\
                                                    "ccp_alpha":0.0,"max_samples":0.9},\
                  'TweedieRegressor':{"alpha":10},\
                  'SVR':{"C":1.2},\
                  'XGBRegressor':{'eta':0.1,'gamma':10,'max_depth':4,'min_child_weight':0,\
                                   'lambda':0.005,'n_estimators':100,'colsample_bytree':0.3,'subsample':0.9,"scale_pos_weight":1},\
                  'LGBMRegressor':{'num_leaves':5,'n_estimators':100,'reg_alpha':0.1,'learning_rate':0.01,\
                            'min_child_samples':15,'subsample':0.9,'colsample_bytree':0.7}}
        
    pipe_grid_classifier = {'AdaBoostClassifier':{"Scaler":None,"Features":"all",\
                                                  "feature selection":SelectPercentile(score_func = f_classif, percentile = 2)},\
                  'BaggingClassifier':{"Scaler":None,"Features":"all"},\
                  'GaussianProcessClassifier':{"Scaler":"normal","Features":"numeric",\
                                               "feature selection":SelectPercentile(score_func = f_classif, percentile = 4)},\
                  'GradientBoostingClassifier':{"Scaler":None,"Features":"all",\
                                                "feature selection":SelectPercentile(score_func = f_classif, percentile = 50)},\
                  'HistGradientBoostingClassifier':{"Scaler":None,"Features":"all",\
                                                    "feature selection":SelectPercentile(score_func = f_classif, percentile = 10)},\
                  'LinearDiscriminantAnalysis':{"Scaler":None,"Features":"numeric",\
                                                "feature selection":SelectPercentile(score_func = f_classif, percentile = 20)},\
                  'LinearSVC':{"Scaler":"normal","Features":"numeric"},\
                  'LogisticRegression':{"Scaler":"normal","Features":"numeric",\
                                        "feature selection":SelectPercentile(score_func = f_classif, percentile = 20)},\
                  'MLPClassifier':{"Scaler":"normal","Features":"numeric"},\
                  'RandomForestClassifier':{"Scaler":None,"Features":"all",\
                                            "feature selection":SelectPercentile(score_func = f_classif, percentile = 2)},\
                  'XGBClassifier':{"Scaler":None,"Features":"all",\
                                   "feature selection":SelectPercentile(score_func = f_classif, percentile = 4)},\
                  'LGBMClassifier':{"Scaler":None,"Features":"all",\
                                    "feature selection":SelectPercentile(score_func = f_classif, percentile = 4)}}
    
    param_grid_classifier = {'AdaBoostClassifier':{"n_estimators":1000,"learning_rate":0.01},\
                  'BaggingClassifier':{"n_estimators":1000, "max_samples": 25, "max_features":10},\
                  'GaussianProcessClassifier':{'kernel':2.0 * RBF(20.0),'max_iter_predict':1000},\
                  'GradientBoostingClassifier':{'learning_rate':0.001,'n_estimators':10000,\
                                                    "min_samples_split":0.05,"min_samples_leaf":0.05,\
                                                    'min_weight_fraction_leaf':0.1,'max_features':"log2","subsample":0.9},\
                  'HistGradientBoostingClassifier':{'learning_rate':0.05,'max_leaf_nodes':10,'max_depth':None,\
                                                    'min_samples_leaf':4,'l2_regularization':1,'max_iter':1000,\
                                                        "interaction_cst":"pairwise"},\
                  'LinearDiscriminantAnalysis':{"solver":"eigen","shrinkage":0.85,'n_components':None,\
                                                'tol':1.0e-3},\
                  'LinearSVC':{"penalty":'l2'},\
                  'LogisticRegression':{"solver":"saga","penalty":"l1","C":0.05,'tol':1e-3,'class_weight':{0:37,1:63}},\
                  'MLPClassifier':{"hidden_layer_sizes":(50,10,5),"activation":"relu","alpha":0.01,\
                                   "learning_rate":"adaptive","momentum":0.99,"early_stopping":True},\
                  'RandomForestClassifier':{"max_depth":4,"min_samples_split":4,"min_samples_leaf":2,\
                                                'max_features':"log2",\
                                                    "class_weight":"balanced","n_estimators":1000},\
                  'XGBClassifier':{'eta':0.05,'gamma':1,'max_depth':4,'min_child_weight':0,\
                                   'lambda':0.1,'n_estimators':1000,'colsample_bytree':0.4,'subsample':0.9,"scale_pos_weight":1},\
                  'LGBMClassifier':{'num_leaves':5,'n_estimators':1000,'reg_alpha':0.1,'learning_rate':0.01,\
                            'min_child_samples':20,'subsample':0.85,'colsample_bytree':0.6},\
                      }

    # first for cpc model
    
    # drop outliers
    features = np.array(features,dtype="float")
    # first set to nan those 5 standard deviations from the mean
    outliers = zscore(features,axis = 0,nan_policy = "omit")
    features[np.abs(outliers) > 5] = np.nan
    # now with 3 standard deviations
    outliers = zscore(features,axis = 0,nan_policy = "omit")
    features[np.abs(outliers) > 3] = np.nan
    
    # imputation method
    imputer = KNNImputer(n_neighbors = 5, weights = "distance").fit(features)
    
    # apply imputation method
    features = imputer.transform(features)
    
    # save paramaters 
    mu = np.nanmean(features,axis = 0);
    sigma = np.nanstd(features,axis = 0)
    parameters = [mu,sigma]
    
    # same for outcome prediction
    # which models are we using
    adaboost = AdaBoostClassifier(); xgb = XGBClassifier(); lgbm = LGBMClassifier()
    logistic = LogisticRegression(); gpc = GaussianProcessClassifier()
    lineardiscriminant = LinearDiscriminantAnalysis()
    gradientboosting= GradientBoostingClassifier(); histgradientboosting = HistGradientBoostingClassifier()
    randomforest = RandomForestClassifier(); mlp = MLPClassifier()


    outcome_model_list = {"AdaBoostClassifier":adaboost,"XGBClassifier":xgb,"LGBMClassifier":lgbm,\
              "GaussianProcessClassifier":gpc,\
              "GradientBoostingClassifier":gradientboosting,\
              "HistGradientBoostingClassifier":histgradientboosting,"RandomForestClassifier":randomforest}
        
    outcome_nmodels = len(outcome_model_list)
    pipe_outcome = []

    for name,model in outcome_model_list.items():
        # check if model has hyperparameters
        # check for hyperparameters
        params = []
        check_param= False
        for key in param_grid_classifier:
            if key == name:
                 print(name)
                 params = param_grid_classifier[key]
                 check_param = True # definitely a better way to do this
                 # add random state if appropriate
                 if "random_state" in model.get_params().keys():
                     params["random_state"] = 23
                     
        # check for scaling options
        scaler_type = []
        check_scale = False
        for key in pipe_grid_classifier:
            if key == name: 
                print(name)
                scaler_type = pipe_grid_classifier[key]
                check_scale = True 
        
        # assign params and normalization technique
        if (check_param) & (check_scale) & (scaler_type["Scaler"] == "normal"):
            pipe = Pipeline(
                steps=[("feature selection",pipe_grid_classifier[name]["feature selection"]),\
                       ("preprocessor", scaler),\
                               (name, model.set_params(**params))]) 
        elif (check_param) & (check_scale) & (scaler_type["Scaler"] == None):
            pipe = Pipeline(
                steps=[("feature selection",pipe_grid_classifier[name]["feature selection"]),\
                       (name, model.set_params(**params))])
        else:
            pipe = Pipeline(
                steps=[(name, model())])
    
        # store all the models in a series of pipes
        pipe_outcome.append((name,pipe))
            
    # initalize and fit the model
    stacking = VotingClassifier(estimators = pipe_outcome,voting = "soft")
    outcome_model = stacking.fit(features,outcomes.ravel())
    
    # get prediction
    y_pred_outcome = outcome_model.predict(features)
    y_pred_conf = np.max(outcome_model.predict_proba(features),axis = 1)
    
    # try also with multiclass model
    stackingmc = VotingClassifier(estimators = pipe_outcome,voting = "soft")
    outcome_model_multiclass = stackingmc.fit(features, cpcs.ravel())
    y_pred_outcome_mc = outcome_model_multiclass.predict(features)
        
    # append outcome prediction to features
    features = np.concatenate([features,y_pred_outcome.reshape(-1,1),\
                               y_pred_conf.reshape(-1,1),y_pred_outcome_mc.reshape(-1,1)],axis = 1)
    
    # which models are we using
    bayesianridge = BayesianRidge(); xgb = XGBRegressor(); lgbm = LGBMRegressor()
    tweedie = TweedieRegressor(); orthogonal = OrthogonalMatchingPursuit()
    svr = SVR(); nusvr = NuSVR(); kneighbors = KNeighborsRegressor();  
    gradientboosting= GradientBoostingRegressor(); histgradientboosting = HistGradientBoostingRegressor()
    randomforest = RandomForestRegressor(); mlp = MLPRegressor()

    cpc_model_list = {"BayesianRidge":bayesianridge,"XGBRegressor":xgb,"LGBMRegressor":lgbm,"GradientBoostingRegressor":gradientboosting,\
              "HistGradientBoostingRegressor":histgradientboosting}

    cpc_nmodels = len(cpc_model_list)
    pipe_cpc = []

    for name,model in cpc_model_list.items():
        # check if model has hyperparameters
        # check for hyperparameters
        params = []
        check_param= False
        for key in param_grid:
            if key == name:
                 print(name)
                 params = param_grid[key]
                 check_param = True # definitely a better way to do this
                 # add random state if appropriate
                 if "random_state" in model.get_params().keys():
                     params["random_state"] = 23
                     
        # check for scaling options
        scaler_type = []
        check_scale = False
        for key in pipe_grid:
            if key == name: 
                print(name)
                scaler_type = pipe_grid[key]
                check_scale = True 
        
        # check for including or not including features
        
        
        # assign params and normalization technique (also add feature selector)
        if (check_param) & (check_scale) & (scaler_type["Scaler"] == "normal"):
            pipe = Pipeline(
                steps=[("feature selection",pipe_grid[name]["feature selection"]),\
                       ("preprocessor", scaler),\
                       (name, model.set_params(**params))]) 
        elif (check_param) & (check_scale) & (scaler_type["Scaler"] == None):
            pipe = Pipeline(
                steps=[("feature selection",pipe_grid[name]["feature selection"]),\
                       (name, model.set_params(**params))])
                       
        else:
            pipe = Pipeline(
                steps=[(name, model())])
    
        # store all the models in a series of pipes
        pipe_cpc.append((name,pipe))
        
    # initalize and fit the model
    stackingcpc = VotingRegressor(estimators = pipe_cpc)
    
    # add output of outcome prediction to features
    cpc_model = stackingcpc.fit(features, cpcs.ravel())
    
    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model,parameters,outcome_model_multiclass)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose): 
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose): # 
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    parameters = models['parameters']
    outcome_model_multiclass = models['outcome_model_multiclass']
    
    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

    # Extract features.
    features = get_features(patient_metadata, recording_metadata, recording_data)
    features = features.reshape(1, -1)
    features = np.array(features,dtype="float")
    
    # set any features that fall outside the training range to nan (drift detection)
    drift = np.abs((features - parameters[0])/parameters[1]) > 3
    features[drift] = np.nan
    
    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = np.max(outcome_model.predict_proba(features))
    outcome_mc = outcome_model_multiclass.predict(features)[0]
    # aggregate model info
    features = np.concatenate([features,np.array([outcome]).reshape(-1,1),\
                               np.array([outcome_probability]).reshape(-1,1),\
                                   np.array([outcome_mc]).reshape(-1,1)],axis = 1)
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)
    
    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model, parameters, outcome_model_multiclass): # see if you can store mu and sigma of training
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model,'parameters':parameters,\
         "outcome_model_multiclass":outcome_model_multiclass}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    rosc = get_rosc(patient_metadata)

    # Combine the patient features.
    patient_features = np.array([age, rosc])

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)
    recording_time = []; feature_len = 2328
    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            if np.sum(signal_data == 0) < 2*30000: # drop if more than an eigth of the array are zeros
                signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
                available_signal_data.append(signal_data)
                recording_time.append(i)

    if len(available_signal_data) > 0:
        available_signal_data = np.stack(available_signal_data,axis = 2)
        (nchan,nt,ntrials) = available_signal_data.shape
        available_signal_data = np.mean(available_signal_data,axis = 2)
        #available_signal_data = np.hstack(available_signal_data) # alternatively stack it via the available data then take average
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
        
        distfeatures = []
        # grab distributions
        distnames = ["norm","expon","dweibull","t","genextreme",\
                                   "gamma","lognorm","uniform","loggamma"]
        # fit for select channels
        fblr = [8,11,12,17]
        for specific_channel in fblr:
            distres = distfit(distr = distnames).fit_transform(available_signal_data[specific_channel,:])
            # get sumary data
            dres = distres["summary"].sort_values("name")[["name","score","loc","scale"]]
            # drop name
            dres.drop(["name"],axis = 1,inplace = True)
            # ravel and store
            distfeatures.append(np.array(dres).ravel())
        
        
        # compute time frequency 
        delta_psd, _ = mne.time_frequency.psd_array_welch(available_signal_data, sfreq=100,  fmin=0.5,  fmax=4.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(available_signal_data, sfreq=100,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(available_signal_data, sfreq=100,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(available_signal_data, sfreq=100, fmin=12.0, fmax=30.0, verbose=False)
        
        # alpha spectral coherence
        spec_con_coh = spectral_connectivity_epochs(available_signal_data.reshape(1,nchan,nt), \
                                                    method='coh', sfreq=100, fmin = 8,fmax = 12,\
                                                        faverage = True)
        
        # construct index for spectral connectivity coherence information
        spec_con_index = []; place = 0
        for channel in reversed(range(1,nchan)):
            spec_con_index.append(place + np.arange(channel + 1,nchan + 1).astype("int"))
            place = spec_con_index[-1][-1] + 1
        spec_con_index = np.hstack(spec_con_index)
        # subset alpha spectral coherence
        temp = spec_con_coh.get_data(); coh = temp[spec_con_index].reshape(-1,)
        
        # alpha directed phase lag index
        spec_con_dpli = spectral_connectivity_epochs(available_signal_data.reshape(1,nchan,nt), \
                                                    method='dpli', sfreq=100, fmin = 8,fmax = 12,\
                                                        faverage = True)
  
        temp = spec_con_dpli.get_data(); dpli = temp[spec_con_index].reshape(-1,)
        
        # compute permutation entropy
        perm_ent = ant.perm_entropy(available_signal_data,normalize = True)
        # spectral entropy
        spect_ent = ant.spectral_entropy(available_signal_data, sf=100, method='welch', normalize=True)
        # kolgomoroz complexity
        kol_comp = epochs_compute_komplexity(available_signal_data,2); kol_comp = kol_comp.reshape(-1,)
        # wsmi
        wsmi,smi = epochs_compute_wsmi(available_signal_data, 3, 100, num_seg = 1)
        # create smi index
        wsmi_index = []; place = 0
        for channel in range(1,nchan):
            wsmi_index.append(place + np.arange(channel,nchan).astype("int"))
            place = wsmi_index[-1][-1] + 1
        wsmi_index = np.hstack(wsmi_index)
        wsmi = wsmi.ravel()[wsmi_index]; smi = smi.ravel()[wsmi_index]
        # lempel ziv
        nchan = available_signal_data.shape[0]
        lziv_comp = np.zeros(nchan,)
        for channel in range(0,nchan):
            # 0s and 1s based on the mean
            Bin = available_signal_data[channel,:] >= \
                np.mean(available_signal_data[channel,:]).reshape(-1,1)
            # binarize the sequence
            LZC = "".join(Bin.astype("int").astype("str").tolist()[0])
            lziv_comp[channel,] = ant.lziv_complexity(LZC, normalize=True)

        quality_score = np.nanmean(get_quality_scores(recording_metadata)) 
        # other recording number features
        rm = np.mean(recording_time); rmax = np.max(recording_time); rmin = np.min(recording_time)
        rlength = np.size(recording_time)  
        
        recording_features = np.hstack((signal_mean, signal_std, delta_psd.ravel(), theta_psd.ravel(), \
                                        alpha_psd.ravel(), beta_psd.ravel(), perm_ent, spect_ent, lziv_comp, \
                                            dpli,coh,kol_comp,wsmi,smi,quality_score,rm,rmax,rmin,rlength,\
                                                np.ravel(distfeatures)))
    
    else:
        recording_features = float('nan') * np.ones(feature_len)


    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    return features

def epochs_compute_komplexity(epochs, nbins, tmin=None, tmax=None,
                              backend='python', method_params=None, num_seg=10, samplingrate=100):
    """Compute complexity (K)

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs on which to compute the wSMI.
    nbins : int
        Number of bins to use for symbolic transformation
    method_params : dictionary.
        Overrides default parameters for the backend used.
        OpenMP specific {'nthreads'}
    backend : {'python', 'openmp'}
        The backend to be used. Defaults to 'python'.
    """

    freq = samplingrate
    if tmin is not None and tmax is not None:
        new_epoch = epochs[:, tmin*freq:tmax*freq]
    else:
        new_epoch = epochs

    numseg = num_seg
    lengthseg = new_epoch.shape[1] // numseg
    data = new_epoch[:][:numseg * lengthseg].reshape(numseg, new_epoch.shape[0], lengthseg)  # epoch*channel*time

    if backend == 'python':
        start_time = time.time()
        komp = _komplexity_python(data, nbins)
        elapsed_time = time.time() - start_time
        logger.info("Elapsed time {} sec".format(elapsed_time))

    return komp


def _symb_python(signal, nbins):
    """Compute symbolic transform"""
    ssignal = np.sort(signal)
    items = signal.shape[0]
    first = int(items / 10)
    last = items - first if first > 1 else items - 1
    lower = ssignal[first]
    upper = ssignal[last]
    bsize = (upper - lower) / nbins
    # bsize gets 0s when signal is 0 for long period of time
    # if np.where(bsize == 0)[0].size > 0:
    #     bsize[np.where(bsize == 0)[0]] = np.mean(bsize[bsize != 0])

    osignal = np.zeros(signal.shape, dtype=np.uint8)
    maxbin = nbins - 1
    a = np.r_[1.]
    with np.errstate(divide='raise'):
        for i in range(items):
            try:
                tbin = int(signal[i] - lower) / bsize
            except RuntimeWarning:
                tbin = 0
            osignal[i] = ((0 if tbin < 0 else maxbin
                           if tbin > maxbin else tbin) + ord('A'))

    return osignal.tostring()


def _komplexity_python(data, nbins):
    """Compute komplexity (K)"""
    ntrials, nchannels, nsamples = data.shape
    k = np.zeros((nchannels, ntrials), dtype=np.float64)
    for trial in range(ntrials):
        for channel in range(nchannels):
            string = _symb_python(data[trial, channel, :], nbins)
            cstring = zlib.compress(string)
            k[channel, trial] = float(len(cstring)) / float(len(string))

    return k


import math
from itertools import permutations

import numpy as np
from mne.utils import logger, _time_mask

def _get_weights_matrix(nsym):
    """Aux function"""
    wts = np.ones((nsym, nsym))  # 初始化一个nsym * nsym大小的矩阵wts，并将其所有元素赋值为1；
    np.fill_diagonal(wts, 0)  # 将矩阵wts的对角线元素赋值为0，表示同一个符号之间的权重为0；
    wts = np.fliplr(wts)  # 将矩阵wts左右翻转
    np.fill_diagonal(wts, 0)  # 将矩阵wts的对角线元素赋值为0；
    wts = np.fliplr(wts)  # 将矩阵wts左右翻转，并将其作为函数的返回值
    return wts


def epochs_compute_wsmi(epochs, kernel, tau, tmin=None, tmax=None,
                        backend='python', method_params=None, n_jobs='auto', samplingrate=100, num_seg = 100):
    """Compute weighted mutual symbolic information (wSMI)

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs on which to compute the wSMI.
    kernel : int
        The number of samples to use to transform to a symbol
    tau : int
        The number of samples left between the ones that defines a symbol.
    method_params : dictionary.
        Overrides default parameters.
        OpenMP specific {'nthreads'}
    backend : {'python', 'openmp'}
        The backend to be used. Defaults to 'python'.
    """

    freq = samplingrate
    if tmin is not None and tmax is not None:
        new_epoch = epochs[:, tmin*freq:tmax*freq]
    else:
        new_epoch = epochs
    numseg = num_seg
    lengthseg = new_epoch.shape[1]//numseg
    fdata = new_epoch[:][:numseg*lengthseg].reshape(new_epoch.shape[0], lengthseg, numseg)   # channel*time*epoch

    # fdata = epochs

    if backend == 'python':
        logger.info("Performing symbolic transformation")
        sym, count = _symb_wsmi_python(fdata, kernel, tau)
        nsym = count.shape[1]  # number of all symbols
        wts = _get_weights_matrix(nsym)  # weigtht of wsmi
        logger.info("Running wsmi with python...")
        wsmi, smi = _wsmi_python(sym, count, wts)

    return wsmi, smi


def _wsmi_python(data, count, wts):
    """Compute wsmi"""
    # nchannels, nsamples, ntrials = data.shape
    nchannels, nsamples, ntrials = data.shape
    nsymbols = count.shape[1]  # 所有符号的数量
    smi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)
    wsmi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)
    for trial in range(ntrials):     # 对于每个试验、每对通道之间的每个组合，都会计算出每个符号对在数据中出现的次数，
        for channel1 in range(nchannels):  # 这个计数被用于计算SMI和wSMI的分母。SMI的分子使用基本的互信息公式计算，
            for channel2 in range(channel1 + 1, nchannels):  # 而wSMI使用带有加权矩阵的互信息公式进行计算，该加权矩阵将更复杂的符号对归结为更简单的符号对
                pxy = np.zeros((nsymbols, nsymbols))
                for sample in range(nsamples):
                    pxy[data[channel1, sample, trial],
                        data[channel2, sample, trial]] += 1   # 计算概率分布pxy
                pxy = pxy / nsamples
                for sc1 in range(nsymbols):
                    for sc2 in range(nsymbols):
                        if pxy[sc1, sc2] > 0:
                            aux = pxy[sc1, sc2] * np.log(
                                pxy[sc1, sc2] / (count[channel1, sc1, trial]*count[channel2, sc2, trial]))  # MI
                               # 遍历每个符号(sc1, sc2)，计算它们的互信息，联合分布
                               # count[channel1, sc1, trial]是通道channel1出现sc1的次数
                            smi[channel1, channel2, trial] += aux
                            wsmi[channel1, channel2, trial] += \
                                (wts[sc1, sc2] * aux)
    wsmi = wsmi / np.log(nsymbols)  # 归一化
    smi = smi / np.log(nsymbols)   # 归一化
    return wsmi, smi

def _define_symbols(kernel):
    result_dict = dict()
    total_symbols = math.factorial(kernel)  # 阶乘
    cursymbol = 0
    for perm in permutations(range(kernel)):   # 遍历所有可能的排列
        order = ''.join(map(str, perm))  # 将排列顺序转换成一个字符串
        if order not in result_dict:
            result_dict[order] = cursymbol
            cursymbol = cursymbol + 1
            result_dict[order[::-1]] = total_symbols - cursymbol  # order[::-1]逆序，简化运算量
    result = []
    for v in range(total_symbols):
        for symbol, value in result_dict.items():
            if value == v:
                result += [symbol]  # 将符号结果放入result，符号symbol为字符串，value为对应值0 1 2 3 4 5
    return result


# Performs symbolic transformation accross 1st dimension
def _symb_wsmi_python(data, kernel, tau):
    """Compute symbolic transform"""
    symbols = _define_symbols(kernel)  # 所有可能的符号
    dims = data.shape  # 输出数据大小  （18,300,100）

    signal_sym_shape = list(dims)  # 转换成list
    signal_sym_shape[1] = data.shape[1] - tau * (kernel - 1)  # 一共有多少个符号
    signal_sym = np.zeros(signal_sym_shape, np.int32)  # signal_sym是一个三维数据（18,300,100）数组，但第二维度变化为（18，300-8*2，100）

    count_shape = list(dims)
    count_shape[1] = len(symbols)  # （18，3！,100）所有可能符号的数量
    count = np.zeros(count_shape, np.int32)

    for k in range(signal_sym_shape[1]):  # 遍历所有符号
        subsamples = range(k, k + kernel * tau, tau)  # 长度为kernel的整数序列
        ind = np.argsort(data[:, subsamples], 1)  # 对信号的所有样本在这个信号片段内进行排序，返回的是每个样本的排序索引
        signal_sym[:, k, ] = np.apply_along_axis(
            lambda x: symbols.index(''.join(map(str, x))), 1, ind)   # symbols.index表示将这一行的排序索引转换成一个符号，join(map(str, x))将排序索引转换成一个字符串

    count = np.double(np.apply_along_axis(
        lambda x: np.bincount(x, minlength=len(symbols)), 1, signal_sym))  # apply_along_axis对符号序列进行行操作，bincount对每行的符号序列进行计数

    return signal_sym, (count / signal_sym_shape[1])
