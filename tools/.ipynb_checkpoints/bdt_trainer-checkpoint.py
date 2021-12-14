import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import xgboost as xg
from sklearn.model_selection import GridSearchCV

from hyperopt import hp
from hyperopt import fmin, tpe, rand,STATUS_OK, STATUS_FAIL,space_eval,Trials

import xgboost as xgb
from sklearn.metrics import mean_squared_error



import matplotlib.pyplot as plt



class train_bdt():
    def __init__(self,data,signal,background,queryString,bdt_vars,params):
        self.sig_name = signal[0]
        self.bkg_name = background[0]

        
        self.df,self.signal,self.background = self.prepareSB(data,signal,background,queryString)
        self.bdt_vars = bdt_vars

        trainFrac = 0.8
        validFrac = 0.1


        np.random.seed(1234)
        theShape = self.df.shape[0]
        ShuffleDF = np.random.permutation(theShape)
        TrainLimit = int(theShape*trainFrac)
        ValidLimit = int(theShape*(trainFrac+validFrac))


        X = self.df[self.bdt_vars].values
        Y = self.df['proc'].values
        weight = self.df['weight'].values


        #Shuffle stuff around
        X = X [ShuffleDF]
        Y = Y [ShuffleDF] 
        weight = weight [ShuffleDF] 


        #Splitting for training and validation
        TrainX,ValidX,TestX = np.split(X,[TrainLimit,ValidLimit])
        TrainY,ValidY,TestY = np.split(Y,[TrainLimit,ValidLimit])
        TrainW,ValidW,TestW = np.split(weight,[TrainLimit,ValidLimit])

        self.TrainX,self.ValidX,self.TestX = TrainX,ValidX,TestX
        self.TrainY,self.ValidY,self.TestY = TrainY,ValidY,TestY
        self.TrainW,self.ValidW,self.TestW = TrainW,ValidW,TestW        


        training = xg.DMatrix(TrainX,label = TrainY, weight = TrainW,feature_names = self.bdt_vars)
        testing  = xg.DMatrix(TestX, label = TestY , weight = TestW, feature_names = self.bdt_vars)
        validating  = xg.DMatrix(ValidX, label = ValidY , weight = ValidW, feature_names = self.bdt_vars)

        self.training_data = training
        self.testing_data = testing
        self.validating_data = validating
        self.params = params



  


        pass


    def AUC(self):

        print ('Default training performance:')
        print ('area under roc curve for training set = %1.3f'%( roc_auc_score(self.TrainY, self.PredY_train, sample_weight=self.TrainW) ) )
        print ('area under roc curve for test set     = %1.3f'%( roc_auc_score(self.TestY,  self.PredY_test,       sample_weight=self.TestW) ) )
        print ('area under roc curve for validity set     = %1.3f'%( roc_auc_score(self.ValidY,  self.PredY_valid,       sample_weight=self.ValidW) ) )
        pass




    def ROC(self):
        #Plotting
        # plt.figure(1)
        f,ax = plt.subplots(figsize=(8,8))

        bkgEff, sigEff, nada = roc_curve(self.TestY, self.PredY_test, sample_weight=self.TestW)
        ax.plot(bkgEff, sigEff,label='Test')
        bkgEff, sigEff, nada = roc_curve(self.TrainY, self.PredY_train, sample_weight=self.TrainW)
        ax.plot(bkgEff, sigEff,label='Train')
        bkgEff, sigEff, nada = roc_curve(self.ValidY, self.PredY_valid, sample_weight=self.ValidW)
        ax.plot(bkgEff, sigEff,label='Valid')

        plt.legend(loc='lower right')

        plt.xlabel('Background efficiency')
        plt.ylabel('Signal efficiency')

    def MVA_plt(self):
        f,ax = plt.subplots(figsize=(8,8))
        binx = np.linspace(0.0,1,50)
        legend_kargs = {
                            # "bbox_to_anchor":(.5, 1),
                            "loc":'upper left',
                            "borderpad":1,
                            "handletextpad":1,
                            "fontsize":11,
                            "labelspacing":1,
                            "fancybox":True
                        }
                            
        _=ax.hist(self.get_mva(self.signal)
            ,bins=binx
            ,linewidth=2
            ,histtype='step'
            ,label=self.sig_name
            ,color='green'
            ,weights = self.signal.weight
            ,density=1
            )

        _=ax.hist(self.get_mva(self.background)
            ,bins=binx
            ,linewidth=2
            ,histtype='step'
            ,label=self.bkg_name
            ,color='red'
            ,weights = self.background.weight
            ,density=1
            )

        _=plt.xlabel(r"ANOM MVA",fontsize=20)
        _=plt.ylabel(r"Events",fontsize=15)
        # _=plt.axvline(125,color='red',linestyle="--")
        _=plt.legend(**legend_kargs)
        # _=plt.text(140, 0.25, "Weighted",fontsize=15)
        # _=plt.grid()
        _=plt.ylim(0,20)


    

    def get_mva(self, df):
        X = df[self.bdt_vars].values
        weight = df['weight'].values
        data = xg.DMatrix(X, feature_names=self.bdt_vars,weight = weight)
        return self.model.predict(data)
        
    def prepareSB(self,data,sigs,bkgs,queryString): #Get Background and Signal
        #preparing signal data
        sig_data = []
        for sig in sigs:
            sig_data.append(data[sig])
            
        signal = pd.concat(sig_data,axis=0)
        signal['proc'] = np.ones(signal.shape[0])
        
        bkg_data = []
        for bkg in bkgs:
            bkg_data.append(data[bkg])
            
        background         = pd.concat(bkg_data,axis=0)
        background['proc'] = np.zeros(background.shape[0])
        all_data = pd.concat((signal,background),axis=0)
        all_data = all_data.query(queryString)
        return all_data,signal,background


    def hyperOpt(self,space,max_evals=100):          
        self.trials = Trials()
        def fn(para):
            model = xg.train(para,self.training_data)
#             PredY_train  = model.predict(self.training_data)
#             loss = -1*roc_auc_score(self.TrainY, PredY_train, sample_weight=self.TrainW) 
            PredY_test  = model.predict(self.testing_data)
            loss = -1*roc_auc_score(self.TestY, PredY_test, sample_weight=self.TestW) 
            # print(para['colsample_bytree'])
            return {'loss': loss, 'status':STATUS_OK}

        result = fmin(fn=fn, space=space, algo=tpe.suggest,trials = self.trials ,max_evals=max_evals)
        
        self.HyperResult = space_eval(space,result)
        
        

    def train(self, params='default'):
        if params == 'default':
            params=self.params
        self.model = xg.train(params,self.training_data)
        self.PredY_train  = self.model.predict(self.training_data)
        self.PredY_test   = self.model.predict(self.testing_data)
        self.PredY_valid  = self.model.predict(self.validating_data)


    
    
#     def __getstate__(self):
#         return self.__dict__


def prepareSB(data,sigs,bkgs,queryString):
    #preparing signal data
    sig_data = []
    for sig in sigs:
        sig_data.append(data[sig])
        
    signal = pd.concat(sig_data,axis=0)
    signal['proc'] = np.ones(signal.shape[0])
    
    bkg_data = []
    for bkg in bkgs:
        bkg_data.append(data[bkg])
        
    background         = pd.concat(bkg_data,axis=0)
    background['proc'] = np.zeros(background.shape[0])
    all_data = pd.concat((signal,background),axis=0)
    all_data = all_data.query(queryString)
    return all_data,signal,background