import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Any, Union

def load_df(path: str) -> pd.DataFrame:
    """CSV 파일을 로드하고 전처리"""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def split_df(df: pd.DataFrame, train_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """데이터프레임을 훈련, 검증, 테스트 세트로 분할
    비율 = [train_pct, 0.1, 나머지]
    """
    n = len(df)
    train_idx = int(n * train_pct)
    val_idx = train_idx + int(n * 0.1)
    
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()
    
    return train_df, val_df, test_df

def scale_data(
    train: pd.DataFrame, 
    val: pd.DataFrame, 
    test: pd.DataFrame, 
    method: str = 'Standard'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """스케일링 메서드에 따라 데이터 정규화
    method ∈ {'Standard', 'MinMax', 'Robust'}
    """
    if method == 'Standard':
        scaler = StandardScaler()
    elif method == 'MinMax':
        scaler = MinMaxScaler()
    elif method == 'Robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")
    
    # 훈련 데이터로만 스케일러 학습
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    
    return train_scaled, val_scaled, test_scaled, scaler

def make_windows(
    arr: np.ndarray, 
    lag: int, 
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """시계열 데이터를 윈도우로 변환
    X: (samples, lag, features), y: (samples,)
    """
    X, y = [], []
    for i in range(len(arr) - lag - horizon + 1):
        # 입력 윈도우 (과거 데이터)
        X.append(arr[i:i+lag])
        # 출력 값 (target, Close 가격)
        y.append(arr[i+lag+horizon-1, 0])  # Close 가격 = 첫 번째 열
    
    return np.array(X), np.array(y)

class WindowDataset(Dataset):
    """시계열 윈도우 데이터셋"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

def prepare_data(
    data_path: str,
    lag: int,
    horizon: int,
    train_pct: float,
    scaling_method: str,
    batch_size: int,
    target_col: str = 'Close',
    feature_cols: List[str] = None
) -> Tuple[Dict[str, DataLoader], Any]:
    """데이터 준비 및 DataLoader 생성 통합 함수"""
    # 데이터 로드
    df = load_df(data_path)
    
    # 특성 선택
    if feature_cols is None:
        feature_cols = ['Close', 'Volume']
    
    selected_df = df[feature_cols].copy()
    
    # 데이터 분할
    train_df, val_df, test_df = split_df(selected_df, train_pct)
    
    # 데이터 스케일링
    train_scaled, val_scaled, test_scaled, scaler = scale_data(
        train_df, val_df, test_df, method=scaling_method
    )
    
    # 시계열 윈도우 생성
    X_train, y_train = make_windows(train_scaled, lag, horizon)
    X_val, y_val = make_windows(val_scaled, lag, horizon)
    X_test, y_test = make_windows(test_scaled, lag, horizon)
    
    # DataLoader 생성
    train_dataset = WindowDataset(X_train, y_train)
    val_dataset = WindowDataset(X_val, y_val)
    test_dataset = WindowDataset(X_test, y_test)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    
    return dataloaders, scaler 