"""Dataset for creating time windows from trajectories."""

from typing import List, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# def get_data(set_name):
#     datasets = {
#         "nominal": ["NF", "NF_2"],
#         "validation": [
#             "f_iml_4mm",  # all but 1
#             "f_pic_080",
#             "f_pic_090",
#             "f_pic_095",
#             "f_pic_105",
#             "f_pic_115",  # all but 2
#             "f_pim_085",
#             "f_pim_090",
#             "f_pim_095",
#             "f_pim_110",
#             "f_pim_115",  # all but 2
#             "f_waf_080",
#             "f_waf_090",
#             "f_waf_095",
#             "f_waf_105",
#             "f_waf_115",  # all but 2
#         ],
#         "testing": [
#             "f_iml_6mm",
#             "f_pic_085",
#             "f_pic_110",
#             "f_pim_080",
#             "f_pim_105",
#             "f_waf_085",
#             "f_waf_110",
#         ],
#         "full": [
#             "f_iml_4mm",  # all but 1
#             "f_iml_6mm",
#             "f_pic_080",
#             "f_pic_085",
#             "f_pic_090",
#             "f_pic_095",
#             "f_pic_105",
#             "f_pic_110",
#             "f_pic_115",  # all but 2
#             "f_pim_080",
#             "f_pim_085",
#             "f_pim_090",
#             "f_pim_095",
#             "f_pim_105",
#             "f_pim_110",
#             "f_pim_115",  # all but 2
#             "f_waf_080",
#             "f_waf_085",
#             "f_waf_090",
#             "f_waf_095",
#             "f_waf_105",
#             "f_waf_110",
#             "f_waf_115",  # all but 2 
#         ]
#     }

#     filenames = datasets[set_name]
#     path = "./dxc25liu-ice/data/trainingdata/wltp_"

#     data = []
#     for file in filenames:
#         data.append(pd.read_csv(path + file + ".csv"))
#     return data

def get_data(set_name, add_fault=False):
    datasets = {
        "nominal": ["NF", "NF_2"],
        "validation": [
            "f_iml_4mm",
            "f_pic_080",
            "f_pic_090",
            "f_pic_095",
            "f_pic_105",
            "f_pic_115",
            "f_pim_085",
            "f_pim_090",
            "f_pim_095",
            "f_pim_110",
            "f_pim_115",
            "f_waf_080",
            "f_waf_090",
            "f_waf_095",
            "f_waf_105",
            "f_waf_115",
        ],
        "testing": [
            "f_iml_6mm",
            "f_pic_085",
            "f_pic_110",
            "f_pim_080",
            "f_pim_105",
            "f_waf_085",
            "f_waf_110",
        ],
        "full": [
            "f_iml_4mm",
            "f_iml_6mm",
            "f_pic_080",
            "f_pic_085",
            "f_pic_090",
            "f_pic_095",
            "f_pic_105",
            "f_pic_110",
            "f_pic_115",
            "f_pim_080",
            "f_pim_085",
            "f_pim_090",
            "f_pim_095",
            "f_pim_105",
            "f_pim_110",
            "f_pim_115",
            "f_waf_080",
            "f_waf_085",
            "f_waf_090",
            "f_waf_095",
            "f_waf_105",
            "f_waf_110",
            "f_waf_115",
        ]
    }

    filenames = datasets[set_name]
    path = "./dxc25liu-ice/data/trainingdata/wltp_"

    data = []
    for file in filenames:
        df = pd.read_csv(path + file + ".csv")
        if set_name == "full" and add_fault:
            fault = file.split('_')[1]  # Extract middle part, e.g. "iml"
            df["fault"] = fault
        data.append(df)
    return data


def delete_features(data: List[pd.DataFrame], features: List[str]):
    for df in data:
        df.drop(columns=features, inplace=True)


def get_extdata():
    FAULT_IDX = 120  # sec
    validation_data = get_data("validation")
    n = len(validation_data)

    nom_data = []
    val_data = []
    for i, df in enumerate(validation_data):
        if i <= n:
            nom_data.append(df[df["time"] < FAULT_IDX])
            val_data.append(df[df["time"] >= FAULT_IDX])
        else:
            val_data.append(df)
    # nom_data += get_data("nominal")
    return nom_data, val_data


class IsolationDataset(Dataset):
    """
    Each trajectory is a DataFrame with a 'time' column and sensor/actuator features.
    """

    def __init__(
        self,
        trajectories: List[pd.DataFrame],
        columns_in: List[str],
        columns_out: List[str],
        scaler: Optional[MinMaxScaler] = None,
        fault_index: Optional[List[int]] = None,
    ):
        """
        Args:
            trajectories (List[pd.DataFrame]): A list of trajectories, each with shape (T_i, F+1),
                                               where the first column is 'time' and the others are features.
            time_window (int): Number of time steps per input sample window.
        """
        # Convert dataframe to numpy and concatenate
        # self.fault_index = fault_index
        # fault_labels = []
        # trajs_in_np = []
        # trajs_out_np = []
        # for i, traj in enumerate(trajectories):
        #     trajs_in_np.append(traj[columns_in].to_numpy())
        #     trajs_out_np.append(traj[columns_out].to_numpy())
        #     if fault_index is not None:
        #         fault_labels.append(np.arange(len(traj)) >= fault_index[i])
        # trajs_in_np = np.concatenate(trajs_in_np, axis=0)
        # trajs_out_np = np.concatenate(trajs_out_np, axis=0)
        # if fault_index is not None:
        #     self.fault_labels = np.concatenate(fault_labels, axis=0)
        # # Normalize the input
        # self.scaler = scaler
        # if scaler is None:
        #     self.scaler = MinMaxScaler()
        #     trajs_in_norm = self.scaler.fit_transform(trajs_in_np)
        # else:
        #     trajs_in_norm = self.scaler.transform(trajs_in_np)

        # self.trajs_in = torch.tensor(trajs_in_norm, dtype=torch.float32)
        # self.trajs_out = torch.tensor(trajs_out_np, dtype=torch.float32)
        self.fault_index = fault_index
        fault_labels = []
        trajs_in_np = []
        trajs_out_np = []

        delete_features(trajectories,features=['ambient_pressure', 'ambient_temperature','wastegate_position', 'time'])
        # Normalize all features first
        all_trajs_np = pd.concat(trajectories, ignore_index=True)

        # Fit or apply the scaler on the full data
        self.scaler = scaler
        if scaler is None:
            self.scaler = MinMaxScaler()
            all_trajs_scaled = self.scaler.fit_transform(all_trajs_np)
        else:
            all_trajs_scaled = self.scaler.transform(all_trajs_np)

        all_trajs_scaled_df = pd.DataFrame(all_trajs_scaled, columns=all_trajs_np.columns)
        trajs_in_np = all_trajs_scaled_df[columns_in].to_numpy()
        trajs_out_np = all_trajs_scaled_df[columns_out].to_numpy()

        self.trajs_in = torch.tensor(trajs_in_np, dtype=torch.float32)
        self.trajs_out = torch.tensor(trajs_out_np, dtype=torch.float32)
        if fault_index is not None:
            self.fault_labels = np.concatenate(fault_labels, axis=0)

    def __len__(self) -> int:
        return len(self.trajs_in)

    def get_scaler(self) -> MinMaxScaler:
        """
        Returns the MinMaxScaler used to normalize the trajectories.
        """
        return self.scaler

    def __getitem__(self, idx: int):
        if self.fault_index is None:
            return self.trajs_in[idx], self.trajs_out[idx]
        else:
            return self.trajs_in[idx], self.trajs_out[idx], self.fault_labels[idx]


class WindowDataset(Dataset):
    """
    PyTorch dataset for creating time windows from multiple trajectories,
    with optional fault labeling and feature normalization.

    Each trajectory is a DataFrame with a 'time' column and sensor/actuator features.
    Optionally, a MinMaxScaler can be provided or fit on the training data.
    """

    def __init__(
        self,
        trajectories: List[pd.DataFrame],
        time_window: int,
        fault_index: Optional[List[int]] = None,
        scaler: Optional[MinMaxScaler] = None,
    ):
        """
        Args:
            trajectories (List[pd.DataFrame]): A list of trajectories, each with shape (T_i, F+1),
                                               where the first column is 'time' and the others are features.
            time_window (int): Number of time steps per input sample window.
            fault_index (Optional[List[int]]): List of fault start indices for each trajectory.
                                               If None, no labels are assigned.
            scaler (Optional[MinMaxScaler]): Optional pre-fitted MinMaxScaler. If None, one is fitted on the data.
        """
        self.time_window = time_window
        self.windows = []
        self.labels = []
        self.fault_index = fault_index
        self.scaler = scaler

        # Convert dataframe to numpy and concatenate
        trajs_np = []
        lens = []
        for traj in trajectories:
            traj_np = traj.drop(columns=["time"]).to_numpy()
            trajs_np.append(traj_np)
            lens.append(len(traj_np))
        trajs_concat = np.concatenate(trajs_np, axis=0)

        # Normalize the data
        if scaler is None:
            self.scaler = MinMaxScaler()
            trajs_norm = self.scaler.fit_transform(trajs_concat)
        else:
            trajs_norm = self.scaler.transform(trajs_concat)

        trajs_norm = np.split(trajs_norm, np.cumsum(lens)[:-1])

        # Convert to torch tensors and add windows
        for i, traj in enumerate(trajs_norm):
            traj = torch.tensor(traj, dtype=torch.float32)
            pad_len = time_window - 1
            padded = torch.cat([torch.zeros((pad_len, traj.shape[1])), traj], dim=0)

            for t in range(traj.shape[0]):
                start = t
                end = t + time_window
                self.windows.append(padded[start:end, :])

                if fault_index is not None:
                    label = 0 if end < fault_index[i] else 1
                    self.labels.append(label)

    def get_scaler(self) -> MinMaxScaler:
        """
        Returns the MinMaxScaler used to normalize the trajectories.
        """
        return self.scaler

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        if self.fault_index is None:
            return self.windows[idx]
        else:
            return self.windows[idx], self.labels[idx]
