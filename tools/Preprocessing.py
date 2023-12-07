import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple, List
import tensorflow.keras.backend as K

from sklearn.metrics import roc_auc_score
from configs import PreprocessingConfig
class Preprocessing:
    def __init__(self, conf: PreprocessingConfig):
        self.batch_size = conf.batch_size
        self.seq_len = conf.window_size
        self.start_timestamp = conf.start_timestamp
        self.end_timestamp = conf.end_timestamp
        self.val_ptc = conf.val_ptc
        self.test_ptc = conf.test_ptc
        self.cols = conf.cols
 
    def handle_nan(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.isnull().values.any():
            data.fillna(data.mean(), inplace=True)
        return data

    def filter_by_time(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        timestamp_mask = (data.index >= self.start_timestamp) & (data.index < self.end_timestamp)
        data = data[timestamp_mask]
        selected_columns = data[columns]
        return selected_columns

    def load_data(self, path: str, columns: List[str]) -> pd.DataFrame:
        data = pd.read_csv(path, index_col='ts')
        data = self.handle_nan(data)
        data = self.filter_by_time(data, columns)
        return data

    def data_splitting(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        times = data.index.values
        last_10pct = times[-int(self.test_ptc*len(times))] 
        last_20pct = times[-int(self.val_ptc*len(times))]
        last_10pct_plus_seqlen = times[-int(self.test_ptc*len(times))-self.seq_len]
        last_20pct_plus_seqlen = times[-int(self.val_ptc*len(times))-self.seq_len] 

        train_num = data[data.index < last_20pct]
        val_num = data[(data.index >= last_20pct_plus_seqlen) & (data.index < last_10pct)]
        test_num = data[data.index >= last_10pct_plus_seqlen]

        train_num_df = train_num.copy()
        val_num_df = val_num.copy()
        test_num_df = test_num.copy()


        train_num = train_num.values
        val_num = val_num.values
        test_num = test_num.values
        
        return train_num, val_num, test_num, train_num_df, val_num_df, test_num_df
        
    def create_sequences(self, data: np.ndarray, seq_len: int) -> np.ndarray:
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequence = data[i:i + seq_len]
            sequences.append(sequence)
        return np.array(sequences)

    


    def process_data(self, file: str) -> Tuple[np.ndarray, List[np.ndarray], pd.Series]:
        X_0 = self.load_data(file, self.cols)
        print("Datos cargados, iniciando el procesamiento...")
        c_prices = X_0['c']
        o_prices = X_0['o']
        h_prices = X_0['h']
        l_prices = X_0['l']

        X_0["ts"] = X_0.index
        ts = X_0["ts"]
        X_0 = X_0.drop(["ts"], axis=1)

        cols = self.cols[4:]
        X_0 = X_0[cols]

        train_0, _, _, _, _, _ = self.data_splitting(X_0)
        # En el método process_data, llamar a create_sequences con el tamaño de lote
        train_0 = self.create_sequences(train_0, self.seq_len)

        c_prices_train, _, _, _, _, _ = self.data_splitting(c_prices)
        o_prices_train, _, _, _, _, _ = self.data_splitting(o_prices)
        h_prices_train, _, _, _, _, _ = self.data_splitting(h_prices)
        l_prices_train, _, _, _, _, _ = self.data_splitting(l_prices)
        ts_train, _, _, _, _, _ = self.data_splitting(ts)

        train_prices = [o_prices_train, h_prices_train, l_prices_train, c_prices_train, ts_train]
        print(train_0.shape)
        print("Procesamiento de datos completado.")
        return train_0, train_prices

    
    