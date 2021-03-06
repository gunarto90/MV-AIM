"""
Code by Gunarto Sindoro Njoo
Written in Python 3.5.2 (Anaconda 4.1.1) -- 64bit
Version 1.0.3
2016/12/08 04:30PM
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
soft_classifier         = model_folder + CLASSIFIER

soft_report             = report_folder + SOFTWARE