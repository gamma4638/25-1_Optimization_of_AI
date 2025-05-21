import torch
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scaler: object,
    device: torch.device
) -> Dict[str, float]:
    """모델 평가"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 원래 스케일로 변환
    predictions = scaler.inverse_transform(np.column_stack([predictions, np.zeros_like(predictions)]))[:, 0]
    actuals = scaler.inverse_transform(np.column_stack([actuals, np.zeros_like(actuals)]))[:, 0]
    
    # 메트릭 계산
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return metrics, predictions, actuals

def plot_predictions(
    predictions: np.ndarray,
    actuals: np.ndarray,
    save_path: str = None
) -> None:
    """예측 결과 시각화"""
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual', color='blue')
    plt.plot(predictions, label='Predicted', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 