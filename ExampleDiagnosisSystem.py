import numpy as np
import pandas as pd
import pickle 
import random

import os
import os.path

from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier

from DiagnosisSystemClass import DiagnosisSystemClass

class ExampleDiagnosisSystem(DiagnosisSystemClass):
    def __init__(self):
        
        self.signalIndices = ["Intercooler_pressure", "intake_manifold_pressure", "air_mass_flow", "engine_speed", "throttle_position", "wastegate_position", "injected_fuel_mass"]

    def Initialize(self):
        # Load model
        if os.path.isfile('exampleDiagnosisSystemParameters.obj'):
            filehandler = open('exampleDiagnosisSystemParameters.obj', 'rb')
            loadDS = pickle.load(filehandler)
            self.anomalyDetector = loadDS['anomalyDetector']
            self.faultClassifier = loadDS['faultClassifier']
            print('Example Diagnosis System loaded.')
        else:
            print('exampleDiagnosisSystemParameters.obj do not exist. Training Daagnosis System.');
            self.Train()

    def Input(self, sample):

        X = sample[self.signalIndices]
        
        # fault detection
        detection = self.anomalyDetector.predict(X);
        
        if detection==1:
            isolation = self.faultClassifier.predict_proba(X)
            isolation = np.concatenate((isolation, np.zeros((1,1))),axis=1)
        else:
            isolation = np.zeros((1,5))

        return(detection, isolation)
        
    def Train(self):
        # Load training data
        data = {
        'wltp_NF' : pd.read_csv('./data/trainingdata/wltp_NF.csv'),
        'wltp_f_pic_090' : pd.read_csv('./data/trainingdata/wltp_f_pic_090.csv'),
        'wltp_f_pic_110' : pd.read_csv('./data/trainingdata/wltp_f_pic_110.csv'),
        'wltp_f_pim_080' : pd.read_csv('./data/trainingdata/wltp_f_pim_080.csv'),
        'wltp_f_pim_090' : pd.read_csv('./data/trainingdata/wltp_f_pim_090.csv'),
        'wltp_f_waf_105' : pd.read_csv('./data/trainingdata/wltp_f_waf_105.csv'),
        'wltp_f_waf_110' : pd.read_csv('./data/trainingdata/wltp_f_waf_110.csv'),
        'wltp_f_iml_6mm' : pd.read_csv('./data/trainingdata/wltp_f_iml_6mm.csv')
        }
        
        random.seed(0)
        
        # fault-free data for fault detector
        Xdet = data['wltp_NF'][self.signalIndices]
        Xdet = Xdet[::2]
        
        print('Training 1SVM-based fault detector...')
        anomalyDetector = OneClassSVM(gamma='auto', nu=0.02).fit(Xdet)
        print('Done!')

        # faulty data for fault classifier
        Xfpic = pd.concat([data['wltp_f_pic_090'][self.signalIndices], data['wltp_f_pic_110'][self.signalIndices]], axis=0)
        Xfpic = Xfpic[::2]
        yfpic = 1*np.ones(len(Xfpic))

        Xfpim = pd.concat([data['wltp_f_pim_080'][self.signalIndices], data['wltp_f_pim_090'][self.signalIndices]], axis=0)
        Xfpim = Xfpim[::2]
        yfpim = 2*np.ones(len(Xfpim))
        
        Xfwaf = pd.concat([data['wltp_f_waf_105'][self.signalIndices], data['wltp_f_waf_110'][self.signalIndices]], axis=0)
        Xfwaf = Xfwaf[::2]
        yfwaf = 3*np.ones(len(Xfwaf))

        Xfiml = data['wltp_f_iml_6mm'][self.signalIndices];
        Xfiml = Xfiml[::2]
        yfiml = 4*np.ones(len(Xfiml))

        Xisol = pd.concat([Xfpic, Xfpim, Xfwaf, Xfiml], axis=0)
        yisol = np.concatenate((yfpic, yfpim, yfwaf, yfiml))

        print('Training Random Forest-based fault classifier with 10 trees... ')
        faultClassifier = RandomForestClassifier(n_estimators=10, random_state=0)
        faultClassifier.fit(Xisol, yisol)
        print('Done!')

        DS = {
            'anomalyDetector' : anomalyDetector,
            'faultClassifier' : faultClassifier
        }
        print('Save trained diagnosis system in exampleDiagnosisSystemParameters.obj ...');
        filehandler = open('exampleDiagnosisSystemParameters.obj', 'wb')
        pickle.dump(DS, filehandler)
        print('Done!')