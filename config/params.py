from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class Parameters:
    # 기본 설명
    description: str = "LSTM 기반 주가 예측 모델"
    
    # 데이터 관련 파라미터
    Lag: int = 60  # 시퀀스 길이(과거 데이터 윈도우 크기)
    Horizon: int = 1  # 예측 기간
    filename: str = "berkshire_lstm.csv"
    targetSeries: str = "Close"
    TrainingSetPercentage: float = 0.7
    Scaling: str = "Standard"  # 'Standard', 'MinMax', 'Robust'
    
    # 모델 관련 파라미터
    model_name: str = "LSTM"
    hidden_sizes: List[int] = None
    dropout: float = 0.2
    
    # 학습 관련 파라미터
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    
    # GA 관련 파라미터
    population_size: int = 20
    num_generations: int = 10
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 32]
    
    def __repr__(self) -> str:
        """JSON 스타일 문자열 표현 (로깅용)"""
        params_dict = {
            "description": self.description,
            "Lag": self.Lag,
            "Horizon": self.Horizon,
            "filename": self.filename,
            "targetSeries": self.targetSeries,
            "TrainingSetPercentage": self.TrainingSetPercentage,
            "Scaling": self.Scaling,
            "model_name": self.model_name,
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate
        }
        return json.dumps(params_dict, indent=2)
    
    def to_dict(self) -> dict:
        """파라미터를 dictionary로 변환"""
        return {
            "description": self.description,
            "Lag": self.Lag,
            "Horizon": self.Horizon,
            "filename": self.filename,
            "targetSeries": self.targetSeries,
            "TrainingSetPercentage": self.TrainingSetPercentage,
            "Scaling": self.Scaling,
            "model_name": self.model_name,
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "early_stopping_patience": self.early_stopping_patience
        } 