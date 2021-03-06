

bdt_vars_2 = [
#  'mass',
#  'diphoton_pt',
#  'diphoton_mva',
#  'pho1_pt' ,
 'pho1_eta',
 'pho1_phi',
#  'pho1_energy',
#  'pho1_full5x5_r9',
 'pho1_idmva',
#  'pho1_genMatchType',
#  'pho2_pt',
 'pho2_eta',
 'pho2_phi',
#  'pho2_energy',
#  'pho2_full5x5_r9',
 'pho2_idmva',
 'pho2_genMatchType',
 'pho1_ptOverMgg',
 'pho2_ptOverMgg',
    
  'cosPhiGG',  
  'cosPhiG1_Mu1',
  'cosPhiG1_Ele1', 
  'cosPhiG2_Mu1',
  'cosPhiG2_Ele1',


    
    
 'mu1_pt',
 'mu1_phi',
 'mu1_eta',
 'mu1_energy',
 'mu2_pt',
 'mu2_ph1',
 'mu2_eta',
 'mu2_energy',
 'ele1_pt',
 'ele1_phi',
 'ele1_eta',
 'ele1_energy',
 'ele2_pt',
 'ele2_phi',
 'ele2_eta',
 'ele2_energy',
 'nMuons',
 'nEle',
    
 'dR_Pho1Jet1_wh',
 'dR_Pho1Jet2_wh',
 'dR_Pho2Jet1_wh',
 'dR_Pho2Jet2_wh',
    
    
 'dR_Pho1Ele1_wh',
 'dR_Pho1Ele2_wh',
 'dR_Pho2Ele1_wh',
 'dR_Pho2Ele2_wh',


 
 'dR_Pho1Mu1_wh',
 'dR_Pho1Mu2_wh',
 'dR_Pho2Mu1_wh',
 'dR_Pho2Mu2_wh',
    
    
 'dR_Mu1Jet1_wh',
 'dR_Mu1Jet2_wh',
 'dR_Mu2Jet1_wh',
 'dR_Mu2Jet2_wh',
    

 'dR_Ele1Jet1_wh',
 'dR_Ele1Jet2_wh',
 'dR_Ele2Jet1_wh',
 'dR_Ele2Jet2_wh',
    
    
    
 'njets',
 'jet1_pt',
 'jet1_phi',
 'jet1_eta',
 'jet1_energy',
 'jet2_pt',
 'jet2_phi',
 'jet2_eta',
 'jet2_energy',
 

]


bdt_vars = [
"pho1_eta",
"pho1_phi",
# "pho1_full5x5_r9",
"pho1_idmva",
"pho1_ptOverMgg",
"pho2_eta",
"pho2_phi",
# "pho2_full5x5_r9",
"pho2_idmva",
"pho2_ptOverMgg",
"mu1_pt",
"mu1_phi",
"mu1_eta",
"mu1_energy",
"mu2_pt",
"mu2_phi",
"mu2_eta",
"mu2_energy",
"ele1_pt",
"ele1_phi",
"ele1_eta",
"ele1_energy",
"ele2_pt",
"ele2_phi",
"ele2_eta",
"ele2_energy",
"jet1_pt",
"jet1_phi",
"jet1_eta",
"jet1_energy",
"jet2_pt",
"jet2_phi",
"jet2_eta",
"jet2_energy",
"cosPhiGG",
"cosPhiG1_Mu1",
"cosPhiG1_Ele1",
"cosPhiG2_Mu1",
"cosPhiG2_Ele1",
"dR_Pho1Ele1_wh",
"dR_Pho2Ele1_wh",
"dR_Pho1Mu1_wh",
"dR_Pho2Mu1_wh",
"dR_Pho1Jet1_wh",
"dR_Pho2Jet1_wh",
"dR_Pho1Jet2_wh",
"dR_Pho2Jet2_wh",
"dR_Mu1Jet1_wh",
"dR_Mu1Jet2_wh",
"dR_Ele1Jet1_wh",
"dR_Ele1Jet2_wh"
]


vars_selected_after_bdt_opt = [

     #Photon Variables
     'pho1_eta',
     'pho1_phi',
     'pho1_full5x5_r9',
     'pho1_idmva',
     'pho2_eta',
     'pho2_phi',
     'pho2_full5x5_r9',
     'pho2_idmva',
     'pho1_ptOverMgg',
     'pho2_ptOverMgg',

     
     #Lepton Variables
     'mu1_pt',
     'mu1_phi',
     'mu1_eta', 
     'mu1_energy',
     'ele1_pt',
     'ele1_phi',
     'ele1_eta',
     'ele1_energy',


     #Jet Variables
     'jet1_pt',
     'jet1_phi',
     'jet1_eta',
     'jet1_energy',
     'jet2_pt',
     'jet2_phi',

     #cosPhi vars
     'cosPhiGG',
     'cosPhiG1_Mu1',
     'cosPhiG1_Ele1',
     'cosPhiG2_Mu1',
     'cosPhiG2_Ele1',

     #dR variables
     'dR_Pho1Ele1_wh',
     'dR_Pho2Ele1_wh',
     'dR_Pho1Mu1_wh',
     'dR_Pho2Mu1_wh',
     
     'dR_Pho1Jet1_wh',
     'dR_Pho2Jet1_wh',
     'dR_Pho1Jet2_wh',
     
     'dR_Mu1Jet1_wh',
     'dR_Mu1Jet2_wh',
     'dR_Ele1Jet1_wh'
]



all_vars = [
'candidate_id',
 'weight',
 'CMS_hgg_mass',
 'sigmaMoM_decorr',
 'dZ',
 'centralObjectWeight',
 'mass',
 'diphoton_pt',
 'diphoton_mva',
 'pho1_pt',
 'pho1_eta',
 'pho1_phi',
 'pho1_energy',
 'pho1_full5x5_r9',
 'pho1_idmva',
 'pho1_genMatchType',
 'pho2_pt',
 'pho2_eta',
 'pho2_phi',
 'pho2_energy',
 'pho2_full5x5_r9',
 'pho2_idmva',
 'pho2_genMatchType',
 'pho1_ptOverMgg',
 'pho2_ptOverMgg',
 'mu1_pt',
 'mu1_phi',
 'mu1_eta',
 'mu1_energy',
 'mu2_pt',
 'mu2_ph1',
 'mu2_eta',
 'mu2_energy',
 'ele1_pt',
 'ele1_phi',
 'ele1_eta',
 'ele1_energy',
 'ele2_pt',
 'ele2_phi',
 'ele2_eta',
 'ele2_energy',
 'nMuons',
 'nEle',
 'dR_Pho1Jet1_wh',
 'dR_Pho1Jet2_wh',
 'dR_Pho2Jet1_wh',
 'dR_Pho2Jet2_wh',
 'dR_Pho1Ele1_wh',
 'dR_Pho1Ele2_wh',
 'dR_Pho2Ele1_wh',
 'dR_Pho2Ele2_wh',
 'dR_Pho1Ele3_wh',
 'dR_Pho2Ele3_wh',
 'dR_Pho1Ele4_wh',
 'dR_Pho2Ele4_wh',
 'dR_Pho1Ele5_wh',
 'dR_Pho2Ele5_wh',
 'dR_Pho1Mu1_wh',
 'dR_Pho1Mu2_wh',
 'dR_Pho2Mu1_wh',
 'dR_Pho2Mu2_wh',
 'dR_Mu1Jet1_wh',
 'dR_Mu1Jet2_wh',
 'dR_Mu2Jet1_wh',
 'dR_Mu2Jet2_wh',
 'dR_Mu1Jet3_wh',
 'dR_Mu2Jet3_wh',
 'dR_Mu1Jet4_wh',
 'dR_Mu2Jet4_wh',
 'dR_Mu1Jet5_wh',
 'dR_Mu2Jet5_wh',
 'dR_Mu1Jet6_wh',
 'dR_Mu2Jet6_wh',
 'dR_Mu1Jet7_wh',
 'dR_Mu2Jet7_wh',
 'dR_Mu3Jet1_wh',
 'dR_Mu3Jet2_wh',
 'dR_Mu3Jet3_wh',
 'dR_Mu3Jet4_wh',
 'dR_Mu3Jet5_wh',
 'dR_Mu3Jet6_wh',
 'dR_Mu3Jet7_wh',
 'dR_Mu4Jet1_wh',
 'dR_Mu4Jet2_wh',
 'dR_Mu4Jet3_wh',
 'dR_Mu4Jet4_wh',
 'dR_Mu4Jet5_wh',
 'dR_Mu4Jet6_wh',
 'dR_Mu4Jet7_wh',
 'dR_Mu5Jet1_wh',
 'dR_Mu5Jet2_wh',
 'dR_Mu5Jet3_wh',
 'dR_Mu5Jet4_wh',
 'dR_Mu5Jet5_wh',
 'dR_Mu5Jet6_wh',
 'dR_Mu5Jet7_wh',
 'dR_Ele1Jet1_wh',
 'dR_Ele1Jet2_wh',
 'dR_Ele2Jet1_wh',
 'dR_Ele2Jet2_wh',
 'dR_Ele1Jet3_wh',
 'dR_Ele2Jet3_wh',
 'dR_Ele1Jet4_wh',
 'dR_Ele2Jet4_wh',
 'dR_Ele1Jet5_wh',
 'dR_Ele2Jet5_wh',
 'dR_Ele1Jet6_wh',
 'dR_Ele2Jet6_wh',
 'dR_Ele1Jet7_wh',
 'dR_Ele2Jet7_wh',
 'dR_Ele3Jet1_wh',
 'dR_Ele3Jet2_wh',
 'dR_Ele3Jet3_wh',
 'dR_Ele3Jet4_wh',
 'dR_Ele3Jet5_wh',
 'dR_Ele3Jet6_wh',
 'dR_Ele3Jet7_wh',
 'dR_Ele4Jet1_wh',
 'dR_Ele4Jet2_wh',
 'dR_Ele4Jet3_wh',
 'dR_Ele4Jet4_wh',
 'dR_Ele4Jet5_wh',
 'dR_Ele4Jet6_wh',
 'dR_Ele4Jet7_wh',
 'dR_Ele5Jet1_wh',
 'dR_Ele5Jet2_wh',
 'dR_Ele5Jet3_wh',
 'dR_Ele5Jet4_wh',
 'dR_Ele5Jet5_wh',
 'dR_Ele5Jet6_wh',
 'dR_Ele5Jet7_wh',
 'dR_Pho1Jet3_wh',
 'dR_Pho2Jet3_wh',
 'dR_Pho1Jet4_wh',
 'dR_Pho2Jet4_wh',
 'dR_Pho1Jet5_wh',
 'dR_Pho2Jet5_wh',
 'dR_Pho1Jet6_wh',
 'dR_Pho2Jet6_wh',
 'dR_Pho1Jet7_wh',
 'dR_Pho2Jet7_wh',
 'njets',
 'jet1_pt',
 'jet1_phi',
 'jet1_eta',
 'jet1_energy',
 'jet2_pt',
 'jet2_phi',
 'jet2_eta',
 'jet2_energy',
 'jet3_pt',
 'jet3_phi',
 'jet3_eta',
 'jet3_energy',
 'jet4_pt',
 'jet4_phi',
 'jet4_eta',
 'jet4_energy',
 'jet5_pt',
 'jet5_phi',
 'jet5_eta',
 'jet5_energy',
 'jet6_pt',
 'jet6_phi',
 'jet6_eta',
 'jet6_energy',
 'jet7_pt',
 'jet7_phi',
 'jet7_eta',
 'jet7_energy',
 'wh_mva',
 'wh_ptV',
 'rho',
 'nvtx',
 'event',
 'lumi',
 'processIndex',
 'run',
 'nvtx',
 'npu',
 'puweight'
 ]