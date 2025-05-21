import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Any
import numpy as np
from tqdm import tqdm
import os

from src.model import LSTMNet
from src.utils import save_artifacts, time_it

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    
    for X, y in tqdm(dataloader, desc='Training', leave=False):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """검증"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

@time_it
def train_one_run(args: Dict) -> Tuple[float, Dict[str, float]]:
    """단일 학습 실행 - GA와 단일 실험을 위한 공통 엔트리포인트"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 초기화
    n_features = next(iter(args['dataloaders']['train']))[0].shape[2]
    
    model = LSTMNet(
        n_features=n_features,
        lag=args['Lag'],
        hidden_sizes=args['hidden_sizes'],
        dropout=args['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    
    # Early stopping 설정
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # 학습 루프
    for epoch in range(args['epochs']):
        train_loss = train_epoch(model, args['dataloaders']['train'], criterion, optimizer, device)
        val_loss = validate(model, args['dataloaders']['val'], criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{args['epochs']} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= args['early_stopping_patience']:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    
    # 최적의 모델 상태 복원
    model.load_state_dict(best_model_state)
    
    # 테스트 성능 평가
    test_loss = validate(model, args['dataloaders']['test'], criterion, device)
    
    # 아티팩트 저장
    run_id = args.get("run_id", "default")
    save_dir = os.path.join('saved_models', f'run_{run_id}')
    save_artifacts(model, args['scaler'], args, save_dir)
    
    metrics = {
        'train_loss': train_losses[-1],
        'val_loss': best_val_loss,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    return test_loss, metrics 