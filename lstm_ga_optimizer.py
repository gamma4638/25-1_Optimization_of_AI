import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import random
import time
from copy import deepcopy

# ─── 1. 설정 클래스 (기존 config.py) ──────────────────────────────────
@dataclass
class Config:
    # 데이터 관련
    csv_path: str = 'berkshire_lstm.csv'
    feature_cols: list[str] = field(default_factory=lambda: ['Close', 'Volume', 'Measure'])
    target_col: str = 'Close'
    
    # 모델 하이퍼파라미터 (GA 최적화 대상)
    seq_len: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 128

    # 고정 학습 파라미터
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    num_epochs: int = 50  # GA 실행 시간을 고려해 에폭 수 조정
    patience: int = 5

    # 실행 환경
    device: torch.device = field(default_factory=lambda: 
                                  torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def __post_init__(self):
        self.input_size = len(self.feature_cols)

# ─── 2. 데이터 처리 (기존 data.py) ───────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df.sort_index()

def split_df(df: pd.DataFrame, cfg: Config):
    n = len(df)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * (cfg.train_ratio + cfg.val_ratio))
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_val]
    test_df = df.iloc[n_val:]
    return train_df, val_df, test_df

def create_sequences(features: np.ndarray, target: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(target[i + seq_len])
    return np.stack(X), np.array(y)

def make_datasets(cfg: Config):
    df = load_data(cfg.csv_path)
    tr_df, va_df, te_df = split_df(df, cfg)
    
    feat_scaler = StandardScaler().fit(tr_df[cfg.feature_cols])
    targ_scaler = StandardScaler().fit(tr_df[[cfg.target_col]])
    
    tr_feats = feat_scaler.transform(tr_df[cfg.feature_cols])
    tr_targ = targ_scaler.transform(tr_df[[cfg.target_col]]).flatten()
    va_feats = feat_scaler.transform(va_df[cfg.feature_cols])
    va_targ = targ_scaler.transform(va_df[[cfg.target_col]]).flatten()
    te_feats = feat_scaler.transform(te_df[cfg.feature_cols])
    te_targ = targ_scaler.transform(te_df[[cfg.target_col]]).flatten()

    X_tr, y_tr = create_sequences(tr_feats, tr_targ, cfg.seq_len)
    X_va, y_va = create_sequences(va_feats, va_targ, cfg.seq_len)
    X_te, y_te = create_sequences(te_feats, te_targ, cfg.seq_len)

    train_ds = TimeSeriesDataset(X_tr, y_tr)
    valid_ds = TimeSeriesDataset(X_va, y_va)
    test_ds = TimeSeriesDataset(X_te, y_te)
    
    return train_ds, valid_ds, test_ds, feat_scaler, targ_scaler

# ─── 3. 모델 정의 (기존 model.py) ─────────────────────────────────────────
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

def build_model(cfg: Config) -> nn.Module:
    model = LSTMRegressor(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout
    )
    return model.to(cfg.device)

# ─── 4. 학습 및 평가 함수 (기존 train.py, evaluate.py) ────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
    return running_loss / len(loader.dataset)

def train_model(cfg: Config, train_ds: Dataset, valid_ds: Dataset) -> tuple[nn.Module, float]:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False)
    
    model = build_model(cfg).to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_loss = validate_epoch(model, valid_loader, criterion, cfg.device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.patience:
            # print(f'Early stopping at epoch {epoch}')
            break
            
    model.load_state_dict(best_model_state)
    return model, best_val_loss

def evaluate_model(model: nn.Module, test_ds: Dataset, targ_scaler: StandardScaler, cfg: Config):
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    
    preds, trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            preds.append(model(X).cpu().numpy())
            trues.append(y.cpu().numpy())
            
    preds = np.vstack(preds).flatten()
    trues = np.vstack(trues).flatten()

    preds_inv = targ_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    trues_inv = targ_scaler.inverse_transform(trues.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(trues_inv, preds_inv)
    rmse = np.sqrt(mean_squared_error(trues_inv, preds_inv))
    mape = np.mean(np.abs((trues_inv - preds_inv) / (trues_inv + 1e-8))) * 100

    print("\n--- 최종 모델 평가 결과 ---")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAPE: {mape:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(trues_inv, label='Actual')
    plt.plot(preds_inv, label='Predicted', linestyle='--')
    plt.title('Final Model: Actual vs. Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

# ─── 5. 유전 알고리즘 (GA) ───────────────────────────────────────────────
# 하이퍼파라미터 탐색 공간
HPARAM_SPACE = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3, 0.4],
    'lr': [1e-4, 5e-4, 1e-3, 5e-3],
    'seq_len': [30, 60, 90, 120],
    'batch_size': [32, 64, 128]
}

# GA 설정
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
CX_PROB = 0.7  # 교차 확률
MUT_PROB = 0.3 # 변이 확률

def create_individual() -> dict:
    """랜덤 하이퍼파라미터 조합(개체) 생성"""
    return {k: random.choice(v) for k, v in HPARAM_SPACE.items()}

def evaluate_fitness(individual: dict, base_cfg: Config, verbose=True) -> float:
    """
    개체(하이퍼파라미터 셋)의 적합도(검증 손실) 평가
    """
    # 개별 설정을 담을 Config 객체 생성
    cfg = deepcopy(base_cfg)
    for key, value in individual.items():
        setattr(cfg, key, value)
    
    # 데이터셋 생성 (seq_len이 바뀔 수 있으므로 매번 생성)
    try:
        train_ds, valid_ds, _, _, _ = make_datasets(cfg)
    except Exception as e:
        print(f"데이터셋 생성 중 오류 (Hparams: {individual}): {e}")
        return float('inf') # 에러 발생 시 최악의 점수 반환

    if len(train_ds) == 0 or len(valid_ds) == 0:
        return float('inf')

    # 모델 학습 및 검증
    _, val_loss = train_model(cfg, train_ds, valid_ds)
    
    if verbose:
        param_str = ", ".join([f"{k}:{v}" for k,v in individual.items()])
        print(f"  HParams: [{param_str}] -> Val Loss: {val_loss:.6f}")
        
    return val_loss

def tournament_selection(population: list, fitnesses: list, k=3) -> dict:
    """토너먼트 선택"""
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k - 1):
        if fitnesses[ix] < fitnesses[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

def crossover(p1: dict, p2: dict) -> tuple[dict, dict]:
    """단일점 교차"""
    c1, c2 = p1.copy(), p2.copy()
    if random.random() < CX_PROB:
        keys = list(HPARAM_SPACE.keys())
        pt = random.randint(1, len(keys) - 1)
        for i in range(pt, len(keys)):
            key = keys[i]
            c1[key], c2[key] = c2[key], c1[key]
    return c1, c2

def mutate(individual: dict) -> dict:
    """랜덤 변이"""
    if random.random() < MUT_PROB:
        key_to_mutate = random.choice(list(HPARAM_SPACE.keys()))
        current_value = individual[key_to_mutate]
        possible_values = [v for v in HPARAM_SPACE[key_to_mutate] if v != current_value]
        if possible_values:
            individual[key_to_mutate] = random.choice(possible_values)
    return individual

# ─── 6. 메인 실행 로직 ───────────────────────────────────────────────────
def main():
    start_time = time.time()
    
    # 기본 설정 로드
    base_cfg = Config()
    
    # --- 유전 알고리즘 시작 ---
    print("--- Genetic Algorithm for Hyperparameter Optimization ---")
    
    # 1. 초기 집단 생성
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_individual = None
    best_fitness = float('inf')

    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n[Generation {gen}/{NUM_GENERATIONS}]")
        
        # 2. 적합도 평가
        fitnesses = [evaluate_fitness(ind, base_cfg) for ind in population]

        # 현재 세대 최고 기록 갱신
        for i in range(POPULATION_SIZE):
            if fitnesses[i] < best_fitness:
                best_fitness = fitnesses[i]
                best_individual = population[i]
                print(f"  >> New Best Found! Val Loss: {best_fitness:.6f}")

        # 3. 다음 세대 생성
        selected = [tournament_selection(population, fitnesses) for _ in range(POPULATION_SIZE)]
        next_population = []
        for i in range(0, POPULATION_SIZE, 2):
            p1, p2 = selected[i], selected[i+1]
            c1, c2 = crossover(p1, p2)
            next_population.append(mutate(c1))
            next_population.append(mutate(c2))
        population = next_population

    print("\n--- GA Optimization Finished ---")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"Best validation loss: {best_fitness:.6f}")
    print("Best hyperparameters:")
    for key, val in best_individual.items():
        print(f"  - {key}: {val}")

    # --- 최적 하이퍼파라미터로 최종 학습 및 평가 ---
    print("\n--- Training Final Model with Best Hyperparameters ---")
    final_cfg = deepcopy(base_cfg)
    for key, value in best_individual.items():
        setattr(final_cfg, key, value)
    
    # 데이터 다시 로드
    train_ds, valid_ds, test_ds, _, targ_scaler = make_datasets(final_cfg)
    
    # 최종 모델 학습 (더 긴 에폭으로 학습 가능)
    final_cfg.num_epochs = 100 
    final_cfg.patience = 10
    print(f"Final training with config: {final_cfg}")
    final_model, _ = train_model(final_cfg, train_ds, valid_ds)
    
    # 체크포인트 저장
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_ga_model.pth')
    torch.save(final_model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")

    # 최종 모델 평가
    evaluate_model(final_model, test_ds, targ_scaler, final_cfg)


if __name__ == '__main__':
    # 실행 전, CSV 파일 경로를 확인해주세요.
    # 예: cfg = Config(csv_path='data/my_stock_data.csv')
    # 아래 data 폴더와 파일을 생성해야 오류 없이 실행됩니다.
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 샘플 데이터 파일 생성
    if not os.path.exists('data/your_data.csv'):
        print("Creating dummy data file 'data/your_data.csv'...")
        sample_dates = pd.date_range(start='2020-01-01', periods=500)
        sample_data = pd.DataFrame({
            'Date': sample_dates,
            'Close': np.random.rand(500).cumsum() + 50,
            'Volume': np.random.randint(1000, 5000, 500),
            'Measure': np.random.randn(500).cumsum()
        })
        sample_data.to_csv('data/your_data.csv', index=False)
        
    main() 