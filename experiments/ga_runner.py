import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import Dict, List, Tuple
import json
import os

from config.params import Parameters
from src.dataset import prepare_data
from src.model import LSTMNet
from src.train import train_one_run
from src.utils import set_seed, ensure_dir

def create_individual(params: Parameters) -> List:
    """개체 생성"""
    return [
        random.randint(30, 120),  # Lag
        random.randint(1, 5),     # Horizon
        random.uniform(0.5, 0.8),  # TrainingSetPercentage
        random.randint(32, 128),   # hidden_sizes[0]
        random.randint(16, 64),    # hidden_sizes[1]
        random.uniform(0.1, 0.5),  # dropout
        random.randint(16, 64),    # batch_size
        random.uniform(0.0001, 0.01),  # learning_rate
        random.randint(50, 200),   # epochs
        random.randint(5, 20)      # early_stopping_patience
    ]

def evaluate_individual(
    individual: List,
    params: Parameters,
    data_path: str
) -> Tuple[float]:
    """개체 평가"""
    # 하이퍼파라미터 설정
    params.Lag = int(individual[0])
    params.Horizon = int(individual[1])
    params.TrainingSetPercentage = individual[2]
    params.hidden_sizes = [int(individual[3]), int(individual[4])]
    params.dropout = individual[5]
    params.batch_size = int(individual[6])
    params.learning_rate = individual[7]
    params.epochs = int(individual[8])
    params.early_stopping_patience = int(individual[9])
    
    # 데이터 준비
    dataloaders, scaler = prepare_data(
        data_path,
        params.Lag,
        params.Horizon,
        params.TrainingSetPercentage,
        params.Scaling,
        params.batch_size,
        params.targetSeries
    )
    
    # 학습 실행
    args = params.to_dict()
    args['dataloaders'] = dataloaders
    args['scaler'] = scaler
    
    test_loss, _ = train_one_run(args)
    
    return (test_loss,)

def run_ga_optimization(
    data_path: str,
    population_size: int = 20,
    num_generations: int = 10,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.8,
    seed: int = 42
) -> Dict:
    """GA 최적화 실행"""
    set_seed(seed)
    
    # DEAP 설정
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    params = Parameters()
    
    # GA 연산자 등록
    toolbox.register("attr_float", create_individual, params)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_individual, params=params, data_path=data_path)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # GA 실행
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=crossover_rate,
        mutpb=mutation_rate,
        ngen=num_generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    # 최적의 하이퍼파라미터 저장
    best_params = {
        'Lag': int(hof[0][0]),
        'Horizon': int(hof[0][1]),
        'TrainingSetPercentage': hof[0][2],
        'hidden_sizes': [int(hof[0][3]), int(hof[0][4])],
        'dropout': hof[0][5],
        'batch_size': int(hof[0][6]),
        'learning_rate': hof[0][7],
        'epochs': int(hof[0][8]),
        'early_stopping_patience': int(hof[0][9]),
        'best_fitness': hof[0].fitness.values[0]
    }
    
    # 결과 저장
    ensure_dir('experiments/results')
    with open('experiments/results/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    return best_params

if __name__ == '__main__':
    data_path = 'data/berkshire_lstm.csv'
    best_params = run_ga_optimization(data_path)
    print("Best parameters:", best_params) 