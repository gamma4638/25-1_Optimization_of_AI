import torch
import numpy as np
import random
import json
import os
import pickle
import time
import functools
from typing import Dict, Any, Callable

def set_seed(seed: int = 42) -> None:
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_pickle(obj: Any, path: str) -> None:
    """객체를 피클 파일로 저장"""
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    """피클 파일에서 객체 로드"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def time_it(func: Callable) -> Callable:
    """함수 실행 시간을 측정하는 데코레이터"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper

def save_artifacts(
    model: torch.nn.Module,
    scaler: Any,
    args: Dict[str, Any],
    save_dir: str
) -> None:
    """학습 아티팩트 저장"""
    ensure_dir(save_dir)
    
    # 모델 저장
    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
    
    # 스케일러 저장
    save_pickle(scaler, os.path.join(save_dir, 'scaler.pkl'))
    
    # 하이퍼파라미터 저장
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(args, f, indent=4)

def load_artifacts(save_dir: str) -> tuple:
    """학습 아티팩트 로드"""
    model_state = torch.load(os.path.join(save_dir, 'best_model.pth'))
    scaler = load_pickle(os.path.join(save_dir, 'scaler.pkl'))
    with open(os.path.join(save_dir, 'args.json'), 'r') as f:
        args = json.load(f)
    return model_state, scaler, args 