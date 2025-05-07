import numpy as np
import pandas as pd
import pickle 
import random

import os
import os.path
import sys

from DiagnosisSystemClass import DiagnosisSystemClass
from fault_detection import FaultDetection
from isolation import ClassificationIsolator

class VandyDiagnosisSystem(DiagnosisSystemClass):
    def __init__(self):
        
        self.signalIndices = ["Intercooler_pressure", "intercooler_temperature", "intake_manifold_pressure", "air_mass_flow", "engine_speed", "throttle_position", "injected_fuel_mass"]

    def Initialize(self):
        self.fault_detector = FaultDetection()
        self.isolator = ClassificationIsolator(pick_one=True)

    def Input(self, sample):

        X = np.array(sample[self.signalIndices])
        t = np.array(sample['time'])
        
        # fault detection
        detection = self.fault_detector.detect(X, t)
        detection = [detection]
        
        if detection[0]==1:
            isolation = np.array(self.isolator.isolate(sample=X.squeeze(), t=t)).reshape(1, 5)
        else:
            isolation = np.zeros((1,5))

        return(detection, isolation)
        