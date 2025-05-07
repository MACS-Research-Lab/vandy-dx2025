import optuna
import numpy as np
import joblib
import torch

from models import TransformerAutoencoder
from persistence import FaultRule

class FaultDetection():
    def __init__(self):
        self.load()
        self.prev_t = None
        self.data_window = None
        self.fault_rule = FaultRule(required_count=140, trigger_percent=0.8)
        
    def detect(self, sample, t):
        if self.prev_t is not None and t < self.prev_t:
            self.prev_t = None
            
        if self.prev_t is None:
            self.reset_buffer()
            self.fault_rule.reset()
            
        sample = self.scaler.transform(sample)
        self.add_to_buffer(sample)
        recons = self.model(self.data_window).detach().numpy()
        error = np.abs(recons - sample)
        self.prev_t = t
        
        initial_diagnosis = (error > self.total_thresh).any()
        persistent_diagnosis = self.fault_rule.apply(initial_diagnosis)
        return persistent_diagnosis
        
    
    def load(self):
        exp_detect = "fault_detection"
        study_detect = optuna.load_study(study_name=exp_detect , storage=f'sqlite:///hyperparams/{exp_detect }.db')
        trial_detect = study_detect.best_trial
        
        exp_deltathr = "fault_detection_delta_threshold_3"
        study_deltathr = optuna.load_study(study_name=exp_deltathr , storage=f'sqlite:///hyperparams/{exp_deltathr }.db')
        trial_deltathr = study_deltathr.best_trial
        self.trial_deltathr = trial_deltathr
        
        hp_detect = {}
        for key in ["batch",
                    "lr",
                    "delta_threshold",
                    "d_model",
                    "bottleneck_dim",
                    "nhead",
                    "num_layers",
                    "dim_feedforward",
                    "dropout",
                    "seq_length",
                    "n_epochs"]:
            hp_detect[key] = trial_detect.params.get(key)
        self.hp_detect = hp_detect
        
        hp_deltathr = {}
        for i in range(7):
            key = f"delta_threshold_{i}"
            hp_deltathr[key] = trial_deltathr.params.get(key)
        
        self.scaler = joblib.load("minmax_scaler_3.pkl")
        self.base_line_threshold =  np.load("base_line_threshold_3.npy")
        threshold = []
        for i,base_thres in enumerate(self.base_line_threshold):
            # threshold.append(base_thres) 
            threshold.append(base_thres + hp_deltathr[f"delta_threshold_{i}"]) 
        self.total_thresh = np.array(threshold)

        self.model = TransformerAutoencoder(
            input_dim=7,
            d_model=hp_detect["d_model"],
            bottleneck_dim=hp_detect["bottleneck_dim"],
            nhead=hp_detect["nhead"],
            num_layers=hp_detect["num_layers"],
            dim_feedforward=hp_detect["dim_feedforward"],
            dropout=hp_detect["dropout"],
            seq_len=hp_detect["seq_length"],
        )
        self.model.load_state_dict(torch.load("model_weights_3.pth"))
        self.model.eval()
        self.reset_buffer()
        
    def add_to_buffer(self, sample):
        sample_tensor = torch.from_numpy(sample).float().unsqueeze(0)
        self.data_window = torch.cat([self.data_window[:, 1:], sample_tensor], dim=1)
        
    def reset_buffer(self):
        self.data_window = torch.zeros(1, self.hp_detect["seq_length"], 7)
        