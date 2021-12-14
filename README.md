# vh_anom_bdt

This is a example of generating bdt classifiers and other relevant plots for the VH anomalous analysis. I have created a simple class as a wrapper around XGBoost to train and save relevant models for each VH anomalous sample that is available.  


You can start with the BDT_Trainer_example.ipynb file to get a sense of how to use the wrapper. 

The feature set is kept in tools/bdt_vars.py and the bdt_trainer wrapper is kept in tools/bdt_trainer.py. You can look at some useful methods to use here. At this time, I have not implemented a way to load models back, but this should be fairly straigh forward to implement. 


Root files for 2018RR and 2017UL are kept in data_files/2021_11_09 that includes WH and ZH leptonic tags

methods in tools/ntuple_meta.py converts these root files to pandas data frame. 

the python wrapper requires the following packages:
- numpy
- pandas 
- matplotlib
- hyperopt
- xgboost
- sklearn
- seaborn
- uproot3






