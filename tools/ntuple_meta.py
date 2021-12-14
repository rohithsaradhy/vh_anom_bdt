# Getting all derived variables

import numpy as np
import uproot3 as up



def CosDeltaPhiGammaGamma(df):
    return np.cos(df.pho1_phi - df.pho2_phi)   

def cosDeltaPhi(df,p1='pho1',p2='mu1'):
    if p2=='mu1':
        return np.where(df.nMuons>0,np.cos(df[f"{p1}_phi"] - df[f"mu1_phi"]),-100)
    if p2=="ele1":
        return np.where(df.nEle>0,np.cos(df[f"{p1}_phi"] - df[f"ele1_phi"]),-100)










import glob

def read_data(filename_root = "data_files/WH_Anomalous/*root"):
    dataset_anom = {}
    dataset_anom['zh'] = {}
    dataset_anom['wh'] = {}
    # print('debug mode')
    for file in glob.glob(filename_root):
        key = file.split("/")[-1].split('.')[0]
        data = up.open(f"{file}")
        first = data.keys()[0].decode("utf-8") 
        second = data[first].keys()[0].decode("utf-8")
        third_wh  = data[f"{first}/{second}"].keys()[0].decode("utf-8") 
        third_zh  = data[f"{first}/{second}"].keys()[1].decode("utf-8") 
        df_wh  =data[f'{first}/{second}/{third_wh}'].pandas.df()
        df_zh  =data[f'{first}/{second}/{third_zh}'].pandas.df()



        def rectify_vars(df):
            if ('mu2_ph1' in df.columns):
                df['mu2_phi'] = df.mu2_ph1
            df['cosPhiGG'] = CosDeltaPhiGammaGamma(df)
            df['cosPhiG1_Mu1'] = cosDeltaPhi(df,'pho1','mu1')
            df['cosPhiG1_Ele1'] = cosDeltaPhi(df,'pho1','ele1')
            df['cosPhiG2_Mu1'] = cosDeltaPhi(df,'pho2','mu1')
            df['cosPhiG2_Ele1'] = cosDeltaPhi(df,'pho2','ele1')
            return df

        df_wh = rectify_vars(df_wh)
        df_zh = rectify_vars(df_zh)
        dataset_anom['wh'][key] = df_wh.astype(float)
        dataset_anom['zh'][key] = df_zh.astype(float)

    return dataset_anom

queryString = '(mass>100.) and (mass<180.) and (pho1_idmva>-0.9) and (pho2_idmva>-0.9) and (pho1_ptOverMgg>0.333) and (pho2_ptOverMgg>0.25)'
queryString += 'and ((nMuons == 1) or (nEle == 1) ) '
    
WHanom_label = {
    'SM':"VH Sample",
    'WHiggs0PMToGG_M125':"g1=1",
    'WHiggs0MToGG_M125' :"fa3=1 (g4=1)",
    'WHiggs0Mf05ph0ToGG_M125':"fa3$^{WH}$=0.5,phia3=0 \n(g1=1,g4=0.1236136)",
    'WHiggs0PHToGG_M125':"fa2=2 (g4=1)",
    'WHiggs0PHf05ph0ToGG_M125':"fa2$^{WH}$=0.5, phia2=0 \n(g1=1,g2=0.0998956)",
    'WHiggs0L1ToGG_M125':"fL1=1(g1$_{prime2}$=1)",
    'WHiggs0L1f05ph0ToGG_M125':"fL1$^{WH}$=0.5,phiL=0 \n(g1=1,g1$_{prime2}$=-525.274)"
    
}





anom_labels = {
    'SM':"VH Sample",
    'WHiggs0PMToGG_M125':"g1=1",
    'WHiggs0MToGG_M125' :"fa3=1 (g4=1)",
    'WHiggs0Mf05ph0ToGG_M125':"fa3^{WH}=0.5,phia3=0 (g1=1,g4=0.1236136)",
    'WHiggs0PHToGG_M125':"fa2=2 (g4=1)",
    'WHiggs0PHf05ph0ToGG_M125':"fa2^{WH}=0.5, phia2=0 (g1=1,g2=0.0998956)",
    'WHiggs0L1ToGG_M125':"fL1=1(g1_{prime2}=1)",
    'WHiggs0L1f05ph0ToGG_M125':"fL1^{WH}=0.5,phiL=0 (g1=1,g1_{prime2}=-525.274)",
    
    'ZHiggs0PMToGG_M125':                "g1=1",
    'ZHiggs0MToGG_M125':                 "fa3=1 (g4=1)",
    'ZHiggs0Mf05ph0ToGG_M125':           "fa3^{ZH}=0.5,phia3=0 (g1=1,g4=0.144057)",
    'ZHiggs0PHToGG_M125':                "fa2=2 (g2=1)",
    'ZHiggs0PHf05ph0ToGG_M125':          "fa2$^{ZH}$=0.5, phia2=0 \n(g1=1,g2=0.112481)",
    'ZHiggs0L1ToGG_M125':                "fL1=1(g1_{prime2}=1)",
    'ZHiggs0L1f05ph0ToGG_M125':          "fL1$^{ZH}$=0.5,phiL=0 \n(g1=1,g1$_{prime2}$=-517.788)",
    'ZHiggs0L1ZgToGG_M125':              "fL1Zg=1 (ghzgs1_{prime2}=1)",
    'ZHiggs0L1Zgf05ph0ToGG_M125':        "fL1Zg^{ZH}=0.5,phiL1Zg=0 \n(g1=1, ghzgs1_{prime2}=-642.9534)",
}


WHanom_color = {
    'SM':"lime",
    'WHiggs0PMToGG_M125':"r",
    'WHiggs0MToGG_M125' :"g",
    'WHiggs0Mf05ph0ToGG_M125':"b",
    'WHiggs0PHToGG_M125':"c",
    'WHiggs0PHf05ph0ToGG_M125':"m",
    'WHiggs0L1ToGG_M125':"y",
    'WHiggs0L1f05ph0ToGG_M125':"k"
}


ZHanom_label = {
    'ZHiggs0PMToGG_M125':                "g1=1",
    'ZHiggs0MToGG_M125':                 "fa3=1 (g4=1)",
    'ZHiggs0Mf05ph0ToGG_M125':           "fa3^{ZH}=0.5,phia3=0 (g1=1,g4=0.144057)",
    'ZHiggs0PHToGG_M125':                "fa2=2 (g2=1)",
    'ZHiggs0PHf05ph0ToGG_M125':          "fa2$^{ZH}$=0.5, phia2=0 \n(g1=1,g2=0.112481)",
    'ZHiggs0L1ToGG_M125':                "fL1=1(g1_{prime2}=1)",
    'ZHiggs0L1f05ph0ToGG_M125':          "fL1$^{ZH}$=0.5,phiL=0 \n(g1=1,g1$_{prime2}$=-517.788)",
    'ZHiggs0L1ZgToGG_M125':              "fL1Zg=1 (ghzgs1_{prime2}=1)",
    'ZHiggs0L1Zgf05ph0ToGG_M125':        "fL1Zg^{ZH}=0.5,phiL1Zg=0 \n(g1=1, ghzgs1_{prime2}=-642.9534)",
}

ZHanom_color = {
    'SM':"lime",
    'ZHiggs0PMToGG_M125':"r",
    'ZHiggs0MToGG_M125' :"g",
    'ZHiggs0Mf05ph0ToGG_M125':"b",
    'ZHiggs0PHToGG_M125':"c",
    'ZHiggs0PHf05ph0ToGG_M125':"m",
    'ZHiggs0L1ToGG_M125':"y",
    'ZHiggs0L1f05ph0ToGG_M125':"k",
    'ZHiggs0L1ZgToGG_M125':"peru",       
    'ZHiggs0L1Zgf05ph0ToGG_M125': "indigo"
}
