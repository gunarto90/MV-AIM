"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.4
2017/01/05 04:00PM
"""

"""
No logic code here !
This file just declares the variables that are used throughout the other python files.
"""

SOFTWARE                = 'app/'
SPATIAL                 = 'spatial/'
TEMPORAL                = 'temporal/'

CROSS_VALIDATION        = 'cv/'
STATISTICS              = 'stats/'
CLASSIFIER              = 'classifier/'
CACHE                   = 'cache/'

dataset_folder          = 'dataset/'
working_folder          = 'working/'
log_folder              = 'log/'
model_folder            = 'model/'
report_folder           = 'report/'

software_folder         = working_folder + SOFTWARE
spatial_folder          = working_folder + SPATIAL
temporal_folder         = working_folder + TEMPORAL

soft_cv_model_folder    = model_folder + SOFTWARE + CROSS_VALIDATION
soft_statistics_folder  = model_folder + SOFTWARE + STATISTICS
soft_classifier         = model_folder + SOFTWARE + CLASSIFIER
soft_users_cache        = model_folder + SOFTWARE + CACHE
soft_users_time_cache   = model_folder + SOFTWARE + CACHE + TEMPORAL

soft_report             = report_folder + SOFTWARE