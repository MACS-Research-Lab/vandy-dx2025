import numpy as np
import torch
import torch.nn.functional as F
import joblib
from opt_hyperparams import FaultRule, IsolationRule
from models import MLP, MLPClassifierFromErrors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Isolation:
    def __init__(self, persistent=True):
        self.prev_t = None
        self.solved = False
        self.load()
        self.persistent = persistent

        count = 120
        trigger_percent = 0.8
        
        if self.persistent:
            self.rule1 = FaultRule(required_count=count, trigger_percent=trigger_percent)  
            self.rule2 = FaultRule(required_count=count, trigger_percent=trigger_percent)
            self.rule3 = FaultRule(required_count=count, trigger_percent=trigger_percent)
            self.rule4 = FaultRule(required_count=count, trigger_percent=trigger_percent)
        else:
            self.rule1 = IsolationRule(required_count=count, trigger_percent=trigger_percent)  
            self.rule2 = IsolationRule(required_count=count, trigger_percent=trigger_percent)
            self.rule3 = IsolationRule(required_count=count, trigger_percent=trigger_percent)
            self.rule4 = IsolationRule(required_count=count, trigger_percent=trigger_percent)

    # Needs to be called even when a fault is not detected yet
    def isolate(self, sample, t):
        if self.prev_t is not None and t < self.prev_t:
            self.prev_t = None

        if self.prev_t is None:
            self.rule1.reset()
            self.rule2.reset()
            self.rule3.reset()
            self.rule4.reset()
            self.solved = False

        sample_np = self.scaler.transform(sample[np.newaxis, :])
        sample_ts = torch.tensor(sample_np, dtype=torch.float32)
        
        model_1_out = self.model[0](sample_ts[:, self.columns1_in]).detach().numpy()
        model_1_error = np.abs(model_1_out - sample_np[:, self.columns1_out])
        model_1_prediction = model_1_error > self.thresh[0]
        model_1_rule = self.rule1.apply(model_1_prediction)

        model_2_out = self.model[1](sample_ts[:, self.columns2_in]).detach().numpy()
        model_2_error = np.abs(model_2_out - sample_np[:, self.columns2_out])
        model_2_prediction = model_2_error > self.thresh[1]
        model_2_rule = self.rule2.apply(model_2_prediction)

        model_3_out = self.model[2](sample_ts[:, self.columns3_in]).detach().numpy()
        model_3_error = np.abs(model_3_out - sample_np[:, self.columns3_out])
        model_3_prediction = model_3_error > self.thresh[2]
        model_3_rule = self.rule3.apply(model_3_prediction)

        model_4_out = self.model[3](sample_ts[:, self.columns4_in]).detach().numpy()
        model_4_error = np.abs(model_4_out - sample_np[:, self.columns4_out])
        model_4_prediction = model_4_error > self.thresh[3]
        model_4_rule = self.rule4.apply(model_4_prediction)

        self.prev_t = t
        if self.persistent and self.solved:
            isolation_result = [0, 0, 0, 0, 0]
            isolation_result[self.solved_idx] = 1
            return (isolation_result, [model_1_rule, model_2_rule, model_3_rule, model_4_rule], [model_1_error, model_2_error, model_3_error, model_4_error])

        # order is (f_pic,f_pim,f_waf,f_iml,f_x)
        isolation_result = [0, 0, 0, 0, 0]
        if np.sum([model_1_rule, model_2_rule, model_3_rule, model_4_rule]) > 2:
            isolation_result[4] = 1
            self.solved = True
            self.solved_idx = 4
        else:
            if (
                model_3_rule == 1
                and model_4_rule == 1
                and model_1_rule == 0
                and model_2_rule == 0
            ):
                isolation_result[3] = 1
                self.solved = True
                self.solved_idx = 3
            elif (
                model_1_rule == 1
                and model_2_rule == 1
                and model_3_rule == 0
                and model_4_rule == 0
            ):
                isolation_result[2] = 1
                self.solved = True
                self.solved_idx = 2
            elif (
                model_1_rule == 1
                and model_3_rule == 1
                and model_2_rule == 0
                and model_4_rule == 0
            ):
                isolation_result[1] = 1
                self.solved = True
                self.solved_idx = 1
            elif (
                model_2_rule == 1
                and model_4_rule == 1
                and model_1_rule == 0
                and model_3_rule == 0
            ):
                isolation_result[0] = 1
                self.solved = True
                self.solved_idx = 0
            else:
                if model_3_rule == 1:
                    isolation_result[3] = 0.33
                    isolation_result[1] = 0.33
                    isolation_result[4] = 0.34
                elif model_1_rule == 1:
                    isolation_result[2] = 0.33
                    isolation_result[1] = 0.33
                    isolation_result[4] = 0.34
                elif model_2_rule == 1:
                    isolation_result[2] = 0.33
                    isolation_result[0] = 0.33
                    isolation_result[4] = 0.34
                elif model_4_rule == 1:
                    isolation_result[3] = 0.33
                    isolation_result[0] = 0.33
                    isolation_result[4] = 0.34
                else:
                    isolation_result = [0.2, 0.2, 0.2, 0.2, 0.2]

        return (isolation_result, [model_1_rule, model_2_rule, model_3_rule, model_4_rule], [model_1_error, model_2_error, model_3_error, model_4_error])

    def load(self):
        self.scaler = joblib.load("minmax_scaler_3.pkl")

        # TODO: replace these with loading in the actual model weights and thresholds
        self.model = []
        self.thresh = []
        input_dim = [5, 5, 4, 4]
        for i in range(4):
            mlp = MLP(input_dim=input_dim[i]).to(device)
            mlp.load_state_dict(
                torch.load(f"model_weights_iso{i}.pth", weights_only=True)
            )
            self.model.append(mlp)
            self.thresh.append(np.load(f"base_line_threshold_iso{i}.npy"))

        self.signalIndices = [
            "Intercooler_pressure",
            "intercooler_temperature",
            "intake_manifold_pressure",
            "air_mass_flow",
            "engine_speed",
            "throttle_position",
            "injected_fuel_mass",
        ]

        # TODO: check that these are being processed in the correct order (they should be)
        # self.columns1_in = [
        #     self.signalIndices.index("intake_manifold_pressure"),
        #     self.signalIndices.index("intercooler_temperature"),
        #     self.signalIndices.index("throttle_position"),
        #     self.signalIndices.index("engine_speed"),
        #     self.signalIndices.index("injected_fuel_mass"),
        # ]
        # self.columns1_out = [self.signalIndices.index("air_mass_flow")]
        self.columns1_in = [
            self.signalIndices.index("intercooler_temperature"),
            self.signalIndices.index("throttle_position"),
            self.signalIndices.index("engine_speed"),
            self.signalIndices.index("injected_fuel_mass"),
            self.signalIndices.index("air_mass_flow"),
        ]
        self.columns1_out = [self.signalIndices.index("intake_manifold_pressure")]

        self.columns2_in = [
            self.signalIndices.index("Intercooler_pressure"),
            self.signalIndices.index("intercooler_temperature"),
            self.signalIndices.index("throttle_position"),
            self.signalIndices.index("engine_speed"),
            self.signalIndices.index("injected_fuel_mass"),
        ]
        self.columns2_out = [self.signalIndices.index("air_mass_flow")]

        self.columns3_in = [
            self.signalIndices.index("intercooler_temperature"),
            self.signalIndices.index("throttle_position"),
            self.signalIndices.index("engine_speed"),
            self.signalIndices.index("injected_fuel_mass"),
        ]
        self.columns3_out = [self.signalIndices.index("intake_manifold_pressure")]

        self.columns4_in = [
            self.signalIndices.index("intercooler_temperature"),
            self.signalIndices.index("injected_fuel_mass"),
            self.signalIndices.index("engine_speed"),
            self.signalIndices.index("throttle_position"),
        ]
        self.columns4_out = [self.signalIndices.index("Intercooler_pressure")]

class ClassificationIsolator():
    def __init__(self, pick_one=False):
        self.pick_one = pick_one
        self.model_paths = ["model_weights_iso0.pth", "model_weights_iso1.pth", "model_weights_iso2.pth", "model_weights_iso3.pth"]

        self.classifier = MLPClassifierFromErrors(self.model_paths, error_mode="abs")
        self.classifier.load_state_dict(torch.load("fault_classification_weights.pth"))

        # Set to evaluation mode (if you're going to use it for inference)
        self.classifier.eval()
        self.scaler = joblib.load("minmax_scaler_3.pkl")
        
        self.signalIndices = [
            "Intercooler_pressure",
            "intercooler_temperature",
            "intake_manifold_pressure",
            "air_mass_flow",
            "engine_speed",
            "throttle_position",
            "injected_fuel_mass",
        ]
        
        self.columns1_in = [
            self.signalIndices.index("intercooler_temperature"),
            self.signalIndices.index("throttle_position"),
            self.signalIndices.index("engine_speed"),
            self.signalIndices.index("injected_fuel_mass"),
            self.signalIndices.index("air_mass_flow"),
        ]
        self.columns1_out = [self.signalIndices.index("intake_manifold_pressure")]

        self.columns2_in = [
            self.signalIndices.index("Intercooler_pressure"),
            self.signalIndices.index("intercooler_temperature"),
            self.signalIndices.index("throttle_position"),
            self.signalIndices.index("engine_speed"),
            self.signalIndices.index("injected_fuel_mass"),
        ]
        self.columns2_out = [self.signalIndices.index("air_mass_flow")]

        self.columns3_in = [
            self.signalIndices.index("intercooler_temperature"),
            self.signalIndices.index("throttle_position"),
            self.signalIndices.index("engine_speed"),
            self.signalIndices.index("injected_fuel_mass"),
        ]
        self.columns3_out = [self.signalIndices.index("intake_manifold_pressure")]

        self.columns4_in = [
            self.signalIndices.index("intercooler_temperature"),
            self.signalIndices.index("injected_fuel_mass"),
            self.signalIndices.index("engine_speed"),
            self.signalIndices.index("throttle_position"),
        ]
        self.columns4_out = [self.signalIndices.index("Intercooler_pressure")]
        
    # Needs to be called even when a fault is not detected yet
    def isolate(self, sample, t):
        sample_np = self.scaler.transform(sample[np.newaxis, :])
        sample_ts = torch.tensor(sample_np, dtype=torch.float32)
        
        model1_in = sample_ts[:, self.columns1_in]
        model1_out = sample_ts[:, self.columns1_out]
        
        model2_in = sample_ts[:, self.columns2_in]
        model2_out = sample_ts[:, self.columns2_out]
        
        model3_in = sample_ts[:, self.columns3_in]
        model3_out = sample_ts[:, self.columns3_out]
        
        model4_in = sample_ts[:, self.columns4_in]
        model4_out = sample_ts[:, self.columns4_out]
        
        # {0: 'iml', 1: 'pic', 2: 'pim', 3: 'waf'} -> needs to be (pic,pim,waf,iml,f_x)
        probs = self.classifier(model1_in, model1_out, model2_in, model2_out, model3_in, model3_out, model4_in, model4_out).detach().numpy().squeeze()
        probs = probs[[1, 2, 3, 0]]
        
        if np.all(probs <= 0.35):
            return [0, 0, 0, 0, 1]
        
        if self.pick_one:
            predicted_index = np.argmax(probs)
            one_hot = np.zeros(4, dtype=float)
            one_hot[predicted_index] = 1.0
            return list(one_hot) + [0]
        else:
            return list(probs) + [0]