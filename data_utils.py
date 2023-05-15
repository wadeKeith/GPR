import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def add_noise(x, noise_sigma=0.0):
    y = x + np.random.normal(0, noise_sigma, size=x.shape)
    return y


def get_data(file_path: str):
    df = pd.read_csv(file_path, encoding="gbk", low_memory=False)
    x = df[["v_Vel", "v_Acc"]].values[:-1]
    y = df["v_Acc"].values[1:]
    return x, y




if __name__ == "__main__":
    x, y = get_data("1 (1334).csv")
    # gpr_data_train_set = get_data("1 (1334).csv")