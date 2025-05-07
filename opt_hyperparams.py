"""Hyper-parameter optimization"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import optuna
from tqdm import tqdm

from models import TransformerAutoencoder, MLP
from dataset import WindowDataset, get_data

BEST_MODEL_PATH = "best_model.pt"
F_IDX = 20 * 120
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def train_isomodel(loader, input_dim, hyperparams, tqdm_disable=True):
    model = MLP(input_dim=input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])

    n_epochs = hyperparams["n_epochs"]

    for epoch in tqdm(range(n_epochs), disable=tqdm_disable):
        epoch_loss = 0.0
        batch_count = 0

        for x, y in tqdm(loader, disable=tqdm_disable, leave=False):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch + 1}/{n_epochs} - Avg Loss: {avg_loss:.6f}")

    return model



def train_model(train_loader, hyperparams, input_dim=10, tqdm_disable=True):
    model = TransformerAutoencoder(
        input_dim=input_dim,
        d_model=hyperparams["d_model"],
        bottleneck_dim=hyperparams["bottleneck_dim"],
        nhead=hyperparams["nhead"],
        num_layers=hyperparams["num_layers"],
        dim_feedforward=hyperparams["dim_feedforward"],
        dropout=hyperparams["dropout"],
        seq_len=hyperparams["seq_length"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])

    n_epochs = hyperparams["n_epochs"]

    for _ in tqdm(range(n_epochs), disable=tqdm_disable):
        for x in tqdm(train_loader, disable=tqdm_disable):
            x = x.to(device)
            output = model(x)
            loss = criterion(output, x[:, -1, :])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# Score should be ((1-FAR) + TDR) / 2
# FAR: False Alarm Rate - the % of samples where a fault is detected when there is not
# TDR: True Detection Rate - the % of samples where a fault is detected when there is
# Score should be maximized (max of 1)


def validate_model_iso(validate_loader, model, threshold, aggregate=False):
    model.eval()

    tp = fp = total_true = total_false = 0
    for X, y, fault in tqdm(validate_loader, disable=False):
        X = X.to(device)
        fault = fault.numpy()

        y_hat = model(X)

        if aggregate:
            error = torch.square(y - y_hat).detach().cpu().numpy()
            is_anomaly = error.sum(axis=1) > threshold
        else:
            error = torch.abs(y - y_hat).detach().cpu().numpy()
            is_anomaly = (error > threshold).any(axis=1)

        tp += np.sum(np.logical_and(is_anomaly, fault))
        fp += np.sum(np.logical_and(is_anomaly, ~fault))
        total_true += np.sum(fault)
        total_false += np.sum(~fault)

    TDR = tp / (total_true + 1e-8)
    FAR = fp / (total_false + 1e-8)

    print("TDR:", TDR)
    print("FAR:", FAR)

    return ((1 - FAR) + TDR) / 2


def validate_model(validate_loader, model, threshold, aggregate=False):
    model.eval()

    tp = fp = total_true = total_false = 0
    for X, y in tqdm(validate_loader, disable=False):
        X = X.to(device)
        y = y.numpy()

        recons = model(X)

        if aggregate:
            error = torch.square(X[:, -1, :] - recons).detach().cpu().numpy()
            is_anomaly = error.sum(axis=1) > threshold
        else:
            error = torch.abs(X[:, -1, :] - recons).detach().cpu().numpy()
            is_anomaly = (error > threshold).any(axis=1)

        tp += np.sum(np.logical_and(is_anomaly, y == 1))
        fp += np.sum(np.logical_and(is_anomaly, y == 0))
        total_true += np.sum(y)
        total_false += np.sum(y == 0)

    TDR = tp / (total_true + 1e-8)
    FAR = fp / (total_false + 1e-8)

    print("TDR:", TDR)
    print("FAR:", FAR)

    return ((1 - FAR) + TDR) / 2


class FaultRule:
    """Fault Rule class"""

    def __init__(self, required_count: int = 100, trigger_percent: float = 0.8):
        """
        Initialize the hysteresis rule.

        Args:
            required_count (int): Size of the sliding window.
            trigger_percent (float): Minimum proportion of 1s to trigger fault.
        """
        self.required_count = required_count
        self.trigger_percent = trigger_percent
        self.reset()

    def reset(self):
        """Reset internal state for reuse."""
        self.state = 0
        self.count = 0
        self.window = []
        self.len_w = 0
        self.trigger_count = int(self.required_count * self.trigger_percent)

    def apply(self, value: int) -> int:
        """
        Apply one step of the rule to a new input value.

        Args:
            value (int): New boolean value (0 or 1).

        Returns:
            int: Current state (0 = nominal, 1 = faulty)
        """
        self.window.append(value)
        self.count += value
        self.len_w += 1

        if self.len_w > self.required_count:
            removed = self.window.pop(0)
            self.count -= removed
            self.len_w -= 1

        if self.state == 0 and self.count >= self.trigger_count:
            self.state = 1

        return self.state
    
class IsolationRule:
    """Fault Rule class"""

    def __init__(self, required_count: int = 100, trigger_percent: float = 0.8):
        """
        Initialize the hysteresis rule.

        Args:
            required_count (int): Size of the sliding window.
            trigger_percent (float): Minimum proportion of 1s to trigger fault.
        """
        self.required_count = required_count
        self.trigger_percent = trigger_percent
        self.reset()

    def reset(self):
        """Reset internal state for reuse."""
        self.state = 0
        self.count = 0
        self.window = []
        self.len_w = 0
        self.trigger_count = int(self.required_count * self.trigger_percent)

    def apply(self, value: int) -> int:
        """
        Apply one step of the rule to a new input value.

        Args:
            value (int): New boolean value (0 or 1).

        Returns:
            int: Current state (0 = nominal, 1 = faulty)
        """
        self.window.append(value)
        self.count += value
        self.len_w += 1

        if self.len_w > self.required_count:
            removed = self.window.pop(0)
            self.count -= removed
            self.len_w -= 1

        return self.count >= self.trigger_count


def validate_model_withrule(
    model,
    val_data,
    scaler,
    windows_size,
    batch_size,
    threshold,
    required_count=200,
    trigger_percent=0.9,
    aggregate=False,
    online=False,
):
    scores = []
    if online:
        batch_size = 1
    for data in val_data:
        dataset = WindowDataset(
            [data],
            time_window=windows_size,
            fault_index=[F_IDX] * len(val_data),
            scaler=scaler,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        if online:
            score = validate_model_withrule_online(
                loader,
                model,
                threshold,
                required_count=required_count,
                trigger_percent=trigger_percent,
                aggregate=aggregate,
            ).item()
        else:
            score = validate_model_withrule_on_traj(
                loader,
                model,
                threshold,
                required_count=required_count,
                trigger_percent=trigger_percent,
                aggregate=aggregate,
            ).item()
        print(f"Score: {score}")
        scores.append(score)
    return np.array(scores)


def validate_model_withrule_online(
    validate_loader,
    model,
    threshold,
    required_count: int = 100,
    trigger_percent: float = 0.8,
    aggregate=False,
):
    """Validate the model in an online fashion"""
    model.eval()

    tp = fp = total_true = total_false = 0
    anomalies = []
    faults = []
    faultrule = FaultRule(
        required_count=required_count, trigger_percent=trigger_percent
    )
    for X, y in tqdm(validate_loader, disable=False):
        X = X.to(device)
        y = y.numpy()

        recons = model(X)

        if aggregate:
            error = torch.square(X[:, -1, :] - recons).detach().cpu().numpy()
            is_anomaly = error.sum(axis=1) > threshold
        else:
            error = torch.abs(X[:, -1, :] - recons).detach().cpu().numpy()
            is_anomaly = (error > threshold).any(axis=1)

        is_anomaly = faultrule.apply(is_anomaly[0])

        anomalies.append(is_anomaly)
        faults.append(y == 1)
    anomalies = np.array(anomalies)
    faults = np.concatenate(faults)

    tp = np.sum(np.logical_and(anomalies, faults))
    fp = np.sum(np.logical_and(anomalies, ~faults))
    total_true = np.sum(faults)
    total_false = np.sum(~faults)

    TDR = tp / (total_true + 1e-8)
    FAR = fp / (total_false + 1e-8)
    print("TDR:", TDR)
    print("FAR:", FAR)
    return ((1 - FAR) + TDR) / 2


def validate_model_withrule_on_traj(
    validate_loader,
    model,
    threshold,
    required_count: int = 100,
    trigger_percent: float = 0.8,
    aggregate=False,
):
    """Validate the model in an individual trajectory"""
    # if feat_mask is None:
    #     feat_mask = np.ones(len(threshold), dtype=bool)
    # else:
    #     feat_mask = np.array(feat_mask, dtype=bool)
    # threshold = threshold[feat_mask]
    model.eval()

    tp = fp = total_true = total_false = 0
    anomalies = []
    faults = []
    for X, y in tqdm(validate_loader, disable=False):
        X = X.to(device)
        y = y.numpy()

        recons = model(X)

        if aggregate:
            error = torch.square(X[:, -1, :] - recons).detach().cpu().numpy()
            is_anomaly = error.sum(axis=1) > threshold
        else:
            error = torch.abs(X[:, -1, :] - recons).detach().cpu().numpy()
            is_anomaly = (error > threshold).any(axis=1)

        anomalies.append(is_anomaly)
        faults.append(y == 1)
    anomalies = np.concatenate(anomalies)
    faults = np.concatenate(faults)

    anomalies = validate_rule(
        anomalies, required_count=required_count, trigger_percent=trigger_percent
    )
    tp = np.sum(np.logical_and(anomalies, faults))
    fp = np.sum(np.logical_and(anomalies, ~faults))
    total_true = np.sum(faults)
    total_false = np.sum(~faults)

    TDR = tp / (total_true + 1e-8)
    FAR = fp / (total_false + 1e-8)
    print("TDR:", TDR)
    print("FAR:", FAR)
    return ((1 - FAR) + TDR) / 2


def validate_rule(
    bool_array: np.ndarray, required_count: int = 100, trigger_percent: float = 0.8
) -> np.ndarray:
    """
    Apply hysteresis fault logic based on a sliding window:
    - Transition from 0 → 1 if the proportion of 1s in the last `required_count` samples
      is greater than or equal to `trigger_percent`.
    - Once in state 1, it remains until the condition for 0 is met (not implemented here).

    Args:
        bool_array (np.ndarray): Boolean 1D array (0 = nominal, 1 = faulty)
        required_count (int): Size of the sliding window.
        trigger_percent (float): Threshold proportion of 1s in the window to trigger fault.

    Returns:
        np.ndarray: Array of same shape with 0 = nominal, 1 = faulty
    """
    result = []
    state = 0
    count = 0
    window = []
    len_w = 0
    trigger_count = int(required_count * trigger_percent)
    for value in bool_array:
        window.append(value.item())
        len_w += 1
        count += value
        if len_w > required_count:
            last_val = window.pop(0)
            count -= last_val
            len_w -= 1
        if state == 0 and (count >= trigger_count):
            state = 1
        result.append(state)
        # print(f"value {value} count {count}, len_w {len_w}, state {state}")
    return np.array(result)


# def validate_rule(validate_loader, model, threshold, n_fix=5):
#     # Iterate over a trajectory of the data
#     # Keep track of faulty or nominal mode
#     # To switch from a mode you need 5 continual detections of the other mode
#     # Note: this needs to be done for each trajectory (don't blend data from other trajectories)
#     # Then the metrics need to be computed in total over the whole dataset

#     raise NotImplementedError


def compute_baseline_threshold(model, train_loader, pairwise=False):
    error = []
    model.eval()
    for X in tqdm(train_loader, disable=True):
        X = X.to(device)
        recons = model(X)
        error.append(torch.abs(X[:, -1, :] - recons).detach().numpy())
    error = np.concatenate(error, axis=0)
    threshold = np.max(error, axis=0)
    return threshold


def compute_baseline_threshold_iso(model, train_loader, nominal_valid_loader, percentile=95, pairwise=False):
    error = []
    model.eval()
    for X, y in tqdm(train_loader, disable=True):
        X = X.to(device)
        y_hat = model(X)
        error.append(torch.abs(y - y_hat).detach().numpy())

    for X, y in tqdm(nominal_valid_loader, disable=True):
        X = X.to(device)
        y_hat = model(X)
        error.append(torch.abs(y - y_hat).detach().numpy())
    error = np.concatenate(error, axis=0)
    threshold = np.percentile(error, percentile)
    return threshold


def objective(trial):
    hyperparams = {
        "batch": trial.suggest_categorical("batch", [32, 64, 128]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "delta_threshold": trial.suggest_float("delta_threshold", -0.05, 0.05),
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128, 256]),
        "bottleneck_dim": trial.suggest_int("bottleneck_dim", 2, 6),
        "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "dim_feedforward": trial.suggest_categorical(
            "dim_feedforward", [64, 128, 256, 512]
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "seq_length": trial.suggest_categorical("seq_length", [5, 10, 20, 30]),
        "n_epochs": trial.suggest_int("n_epochs", 2, 10),
    }

    train_data = get_data("nominal")
    train_dataset = WindowDataset(train_data, time_window=hyperparams["seq_length"])
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch"], shuffle=True
    )
    scaler = train_dataset.get_scaler()

    validation_data = get_data("validation")

    validation_dataset = WindowDataset(
        validation_data,
        time_window=hyperparams["seq_length"],
        fault_index=[F_IDX] * len(validation_data),
        scaler=scaler,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=hyperparams["batch"], shuffle=False
    )

    model = train_model(train_loader, hyperparams)
    threshold = (
        compute_baseline_threshold(model, train_loader) + hyperparams["delta_threshold"]
    )
    score = validate_model(validation_loader, model, threshold)
    # Save the model if it's the best so far

    # if trial.number > 0 and score > trial.study.best_value:
    #     torch.save(model.state_dict(), BEST_MODEL_PATH)
    #     print(f"✅ Saved new best model with score: {score:.4f}")
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="fault_detection",
        storage="sqlite:///./hyperparams/fault_detection.db",
        load_if_exists=True,
    )  # we need to save this too
    study.optimize(objective, n_trials=100, n_jobs=1)
