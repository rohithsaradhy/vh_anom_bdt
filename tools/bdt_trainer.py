import pandas as pd
import numpy as np
import math
from copy import copy
from .xgboost2tmva import *
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import xgboost as xg
from sklearn.model_selection import GridSearchCV

from hyperopt import hp
from hyperopt import fmin, tpe, rand,STATUS_OK, STATUS_FAIL,space_eval,Trials

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from .ntuple_meta import *


import matplotlib.pyplot as plt


def MetricFunc(S,B,Br=10,std=False):
    if std:
        return S/np.sqrt(S+B)
    else:
        return math.sqrt(2*((S+B+Br)*math.log(1+(S/(B+Br)))-S))


class train_bdt():
    def __init__(self, *args, **kwargs):


        self.data        = kwargs['data']  
        self.sig_name    = kwargs['signal'][0]
        self.bkg_name    = kwargs['background'][0]
        self.queryString = kwargs['queryString']
        self.params      = kwargs['params']
        self.df,self.signal,self.background = self.prepareSB(self.data,[self.sig_name],[self.bkg_name],self.queryString)
        
        if 'prune' in kwargs:
            print("pruning variables with high correlation")
            self.bdt_vars    = self.prune_variables(self.signal,kwargs['bdt_vars'],kwargs['prune'])
        else:
            self.bdt_vars    = kwargs['bdt_vars']
        


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

  


        # pass

    def prune_variables(self,df, init_vars,plot=False):
        cormat = df[init_vars].corr()
        if plot:
            plt.figure(figsize=(32, 12))
            sns.heatmap(cormat,annot=True)
            
        corr = cormat
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.9 or corr.iloc[i,j] <= -0.9 :
                    if columns[j]:
                        columns[j] = False
            
        
        selected_column = df[init_vars].columns[columns]
        
        if plot:
            cormat = df[selected_column].corr()
            plt.figure(figsize=(32, 12))
            sns.heatmap(cormat,annot=True)
            
        return selected_column


    def get_sample_size(self):
        return (self.TrainX.shape[0],self.ValidX.shape[0],self.TestX.shape[0])

    def AUC(self,quite=True):
        self.train_auc = roc_auc_score(self.TrainY, self.PredY_train, sample_weight=self.TrainW)
        self.test_auc  = roc_auc_score(self.TestY,  self.PredY_test,  sample_weight=self.TestW)
        self.valid_auc = roc_auc_score(self.ValidY, self.PredY_valid, sample_weight=self.ValidW)
        self.STXS_mva_auc = roc_auc_score(self.df.proc, self.df.vh_mva, sample_weight=self.df.weight)
        if (not quite):
            print ('Default training performance:')
            print ('area under roc curve for training set = %1.3f'%( self.train_auc) )
            print ('area under roc curve for test set     = %1.3f'%( self.test_auc ) )
            print ('area under roc curve for validity set     = %1.3f'%( self.valid_auc ) )
            print ('area under roc curve for STXS MVA     = %1.3f'%( self.STXS_mva_auc ) )
        pass





    def ROC(self):
        #Plotting
        # plt.figure(1)
        self.train_auc = roc_auc_score(self.TrainY, self.PredY_train, sample_weight=self.TrainW)
        self.test_auc  = roc_auc_score(self.TestY,  self.PredY_test,  sample_weight=self.TestW)
        self.valid_auc = roc_auc_score(self.ValidY, self.PredY_valid, sample_weight=self.ValidW)
        


        f,ax = plt.subplots(figsize=(8,8))

        bkgEff, sigEff, nada = roc_curve(self.TrainY, self.PredY_train, sample_weight=self.TrainW)
        ax.plot(bkgEff, sigEff,label=f'Train:{self.TrainX.shape[0]}; AOC = {round(self.train_auc,3)}')

        bkgEff, sigEff, nada = roc_curve(self.TestY, self.PredY_test, sample_weight=self.TestW)
        ax.plot(bkgEff, sigEff,label=f'Test:{self.TestX.shape[0]}; AOC = {round(self.test_auc,3)}')

        bkgEff, sigEff, nada = roc_curve(self.ValidY, self.PredY_valid, sample_weight=self.ValidW)
        ax.plot(bkgEff, sigEff,label=f'Valid:{self.ValidX.shape[0]}; AOC = {round(self.valid_auc,3)}')


        #WH_MVA
        bkgEff, sigEff, nada = roc_curve(self.df.proc, self.df.vh_mva, sample_weight=self.df.weight)
        vh_mva_aoc = roc_auc_score(self.df.proc, self.df.vh_mva, sample_weight=self.df.weight)        
        ax.plot(bkgEff, sigEff,label=f'STXS MVA AOC={round(vh_mva_aoc,3)}')

        plt.legend(loc='lower right')
        plt.title(f"{self.sig_name }")
        plt.xlabel('Background efficiency')
        plt.ylabel('Signal efficiency')



    def MVA_plt(self, AMS=False):

        f,ax = plt.subplots(figsize=(8,8))
        binx = np.linspace(0.0,1,50)





        def get_mva_local(X,weight):
            data = xg.DMatrix(X, feature_names=self.bdt_vars,weight = weight)
            return self.model.predict(data)


        def test_train_valid(signal= 1):
            test_train_valid = (
                get_mva_local(self.TrainX[self.TrainY == signal],self.TrainW[self.TrainY == signal])
                ,get_mva_local(self.TestX[self.TestY == signal],self.TestW[self.TestY == signal])
                ,get_mva_local(self.ValidX[self.ValidY == signal],self.ValidW[self.ValidY == signal])                
                )

            train  = test_train_valid[0]
            test   = test_train_valid[1]
            valid  = test_train_valid[2]

            test_train_valid_weight = (
                self.TrainW[self.TrainY == signal],
                self.TestW[self.TestY == signal],
                self.ValidW[self.ValidY == signal])



            if signal:
                test_train_valid_colors = (
                                            'green',
                                            "mediumspringgreen",
                                            "lightseagreen"
                                            )
                test_train_valid_labels = (
                                            " Signal Train",
                                            'Signal Test',
                                            "Signal Valid"
                                            )
            else:
                test_train_valid_colors = (
                                            "tomato",
                                            'lightsalmon',
                                            "darkorange"
                                            )
                test_train_valid_labels = (
                                            "Background Train",
                                            'Background Test',
                                            "Background Valid" 
                                            )



            # _=ax.hist(test_train_valid
            #     ,bins=binx
            #     ,linewidth=2
            #     # ,histtype='step'
            #     ,stacked=True
            #     ,label=test_train_valid_labels

            #     ,color=test_train_valid_colors
            #     ,weights = test_train_valid_weight
            #     # ,density=1
            #     )


            counts,bins = np.histogram(
                test,
                bins=binx,
                weights= test_train_valid_weight[1],
                density = 1
            ) 

            counts_err,bins = np.histogram(
                test,
                bins=binx,
                # weights= test_train_valid_weight[1],
            ) 



            if (signal):
                color = 'blue'
            else:
                color = 'red'

            x_points = bins[:-1] + 0.5*(bins[1]-bins[0])
            _=ax.errorbar(
                x_points[counts > 0],
                counts[counts >0],
                fmt='o',
                yerr = 1/(np.sqrt(counts_err[counts >0])),
                label = test_train_valid_labels[1],
                color = color,
                ecolor = 'k',
                capsize = 1.5
            )






        def calc_ams(df):
            def get_ams(df,metric_cut):
                dff = copy(df[df.anom_mva > metric_cut])
                sig = dff[dff.proc==1]
                bkg = dff[dff.proc==0]
                ams = MetricFunc(np.sum(sig.weight),np.sum(bkg.weight),std=False)
                return ams
            X,Y = [],[]
            for i in np.linspace(0,1,50):
                X.append(i)
                Y.append(get_ams(df,i))
            ax2 = ax.twinx()
            max_indx = Y.index(max(Y))
            # ax2.axvline(X[max_indx],
            #             ymin=max(Y)-2,
            #             ymax = max(Y)+2,
            #             color='red'
            # )
            ax2.plot(X,Y,'r-.',linewidth=3,label=f'AMS max = {round(X[max_indx],2)},{round(Y[max_indx],2)}')
            ax2.set_ylabel('AMS')
            ax2.set_ylim(0,20)

            legend_kargs = {
                    "bbox_to_anchor":(.375, .8),
                    # "loc":'upper left',
                    "borderpad":1,
                    "handletextpad":1,
                    "fontsize":11,
                    "labelspacing":1,
                    "fancybox":True
                }
            _=ax2.legend(**legend_kargs)
            # _=ax2.grid()



        
        legend_kargs = {
                            # "bbox_to_anchor":(.5, 1),
                            "loc":'upper left',
                            "borderpad":1,
                            "handletextpad":1,
                            "fontsize":11,
                            "labelspacing":1,
                            "fancybox":True,
                            'ncol':2
                        }
                            


        #Signal



        test_train_valid(signal= 1)
        _=ax.hist(self.get_mva(self.signal)
            ,bins=binx
            ,linewidth=2
            # ,histtype='step'
            ,label=anom_labels[self.sig_name]
            # ,color='darkgreen'
            ,fc=(0,0,1,.5)
            ,weights = self.signal.weight
            ,density=1
            )
        _=ax.hist(self.get_mva(self.signal)
            ,bins=binx
            ,linewidth=2
            ,histtype='step'
            # ,label=WHanom_label[self.sig_name]
            ,color='blue'
            # ,fc=(0,0,1,.5)
            ,weights = self.signal.weight
            ,density=1
            )


        test_train_valid(signal= 0)
        _=ax.hist(self.get_mva(self.background)
            ,bins=binx
            ,linewidth=2
            # ,histtype='step'
            ,label=anom_labels[self.bkg_name]
            # ,color='maroon'
            ,fc=(1,0,0,.5)
            ,weights = self.background.weight
            ,density=1
            )
        _=ax.hist(self.get_mva(self.background)
            ,bins=binx
            ,linewidth=2
            ,histtype='step'
            # ,label=WHanom_label[self.bkg_name]
            ,color='red'
            # ,fc=(1,0,0,.5)
            ,weights = self.background.weight
            ,density=1
            )



        
        _=ax.legend(**legend_kargs)

        if (AMS):
            f,ax = plt.subplots(figsize=(8,8))
            _=ax.hist(self.get_mva(self.background)
                        ,bins=binx
                        ,linewidth=2
                        ,histtype='step'
                        # ,label=WHanom_label[self.bkg_name]
                        ,color='red'
                        # ,fc=(1,0,0,.5)
                        ,weights = self.background.weight
                        )


            calc_ams(self.df)
            _=ax.set_xlabel(r"ANOM MVA",fontsize=20)
            

        _=ax.set_ylabel(r"Events",fontsize=15)
        _=ax.set_xlabel(r"BDT MVA",fontsize=15)

        # _=plt.axvline(125,color='red',linestyle="--")
        # _=plt.legend(**legend_kargs)
        _=plt.title(f"{self.sig_name}")
        # _=plt.text(140, 0.25, "Weighted",fontsize=15)
        # _=plt.grid()
        # _=plt.ylim(0,20)


    

    def get_mva(self, df):
        X = df[self.bdt_vars].values
        weight = df['weight'].values
        data = xg.DMatrix(X, feature_names=self.bdt_vars,weight = weight)
        return self.model.predict(data)

        
    def prepareSB(self,data,sigs,bkgs,queryString): #Get Background and Signal
        #preparing signal data
        sig_data = []
        for sig in sigs:
            print(sig)
            sig_data.append(data[sig])
            
        signal = pd.concat(sig_data,axis=0)
        signal['proc'] = np.ones(signal.shape[0])
        
        bkg_data = []
        for bkg in bkgs:
            bkg_data.append(data[bkg])
            
        background         = pd.concat(bkg_data,axis=0)
        background['proc'] = np.zeros(background.shape[0])
        all_data = pd.concat((signal,background),axis=0)
        # all_data = all_data.query(queryString)
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
        self.df['anom_mva'] = self.get_mva(self.df)


    def train_classifier(self):
        params = self.params
        mdl = xg.XGBClassifier()
        self.model = mdl.fit(self.TrainX,self.TrainY,sample_weight= self.TrainW)
        self.PredY_train  = self.model.predict(self.TrainX)#,sample_weight= self.TrainW)
        self.PredY_test   = self.model.predict(self.TestX)#,sample_weight= self.TestW)
        self.PredY_valid  = self.model.predict(self.ValidX)#, sample_weight= self.ValidW)
        

    # def train(self, params='default'):
    #     if params == 'default':
    #         params=self.params

    #     self.model = xg.

    def plot_feature_importance(self):
        xgb.plot_importance(self.model,title=self.sig_name)
    

    def plot_var(self,var,binx=100,queryString=""):
        f,ax = plt.subplots(figsize=(8,8))
        # binx = 100

        if queryString != "":
            sig = self.signal.query(queryString)[var]
            sig_weight = self.signal.query(queryString)['weight']
            bkg = self.background.query(queryString)[var]
            bkg_weight = self.background.query(queryString)['weight']
        else:
            sig = self.signal[var]
            sig_weight = self.signal['weight']
            bkg = self.background[var]
            bkg_weight = self.background['weight']
            


        _=ax.hist(sig
            ,bins=binx
            ,linewidth=2
            ,histtype='step'
            ,label=WHanom_label[self.sig_name]
            ,color='darkgreen'
            ,weights = sig_weight
            # ,density=1
            )


        _=ax.hist(bkg
            ,bins=binx
            ,linewidth=2
            ,histtype='step'
            ,label=WHanom_label[self.bkg_name]
            ,color='red'
            ,weights = bkg_weight
            # ,density=1
            )
        legend_kargs = {
                # "bbox_to_anchor":(.5, 1),
                "loc":'upper left',
                "borderpad":1,
                "handletextpad":1,
                "fontsize":11,
                "labelspacing":1,
                "fancybox":True,
                # 'ncol':2
            }
        ax.legend(**legend_kargs)
        ax.set_xlabel(var,fontsize=15)
        ax.set_ylabel("Events",fontsize=15)



    
    def save_tmva_xml(self,file_loc=''):
        if file_loc=='':
            print("specify file location")
        else:
            mdl = self.model.get_dump()
            input_vars=[]
            for key in self.bdt_vars:
                input_vars.append((key,'F'))
        # for i in input_vars:
        #     print(i)

            convert_model(mdl,input_variables=input_vars,output_xml=file_loc)




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