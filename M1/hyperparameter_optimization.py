"""
Hyperparameter Optimization for DARU-Net
Uses Optuna for automated hyperparameter tuning
"""

import optuna
import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from train_ultimate_optimized import train_ultimate_model, create_advanced_datasets
from darunet import DARU_Net

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna"""
    
    def __init__(self, n_trials=50, timeout=None):
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.best_score = 0.0
        self.study = None
        
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        
        # Suggest hyperparameters
        config = {
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'num_epochs': 30,  # Reduced for faster optimization
            'patience': 10,
            'use_paper_config': trial.suggest_categorical('use_paper_config', [True, False]),
            'enhanced_complexity': trial.suggest_categorical('enhanced_complexity', [True, False]),
            'input_size': trial.suggest_categorical('input_size', [(256, 256), (512, 512)]),
        }
        
        # Additional loss function parameters
        focal_weight = trial.suggest_float('focal_weight', 0.1, 0.6)
        dice_weight = trial.suggest_float('dice_weight', 0.1, 0.6)
        tversky_weight = trial.suggest_float('tversky_weight', 0.1, 0.4)
        boundary_weight = 1.0 - focal_weight - dice_weight - tversky_weight
        
        if boundary_weight < 0:
            boundary_weight = 0.1
            # Normalize weights
            total = focal_weight + dice_weight + tversky_weight + boundary_weight
            focal_weight /= total
            dice_weight /= total
            tversky_weight /= total
            boundary_weight /= total
        
        config['loss_weights'] = {
            'focal_weight': focal_weight,
            'dice_weight': dice_weight,
            'tversky_weight': tversky_weight,
            'boundary_weight': boundary_weight
        }
        
        try:
            # Train model with suggested parameters
            model, trainer, results = train_ultimate_model(config)
            
            # Return validation F1 score as objective
            return results['best_val_f1']
            
        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            # Return a low score for failed trials
            return 0.0
    
    def optimize(self, study_name="daru_net_optimization"):
        """Run hyperparameter optimization"""
        
        print("🔬 Starting Hyperparameter Optimization")
        print("=" * 50)
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f'sqlite:///M1/results/{study_name}.db',
            load_if_exists=True
        )
        
        # Add custom sampler for better exploration
        self.study.sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
            warn_independent_sampling=False
        )
        
        # Add pruner for early stopping of unpromising trials
        self.study.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        )
        
        print(f"🎯 Running {self.n_trials} trials...")
        start_time = time.time()
        
        # Optimize
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        optimization_time = time.time() - start_time
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"\n🏆 Optimization completed in {optimization_time:.2f} seconds")
        print(f"🎯 Best F1 Score: {self.best_score:.4f}")
        print(f"📋 Best Parameters:")
        for key, value in self.best_params.items():
            print(f"   {key}: {value}")
        
        # Save results
        self.save_results()
        
        return self.best_params, self.best_score
    
    def save_results(self):
        """Save optimization results"""
        results = {
            'best_score': self.best_score,
            'best_params': self.best_params,
            'n_trials': len(self.study.trials),
            'optimization_time': time.time(),
            'study_name': self.study.study_name
        }
        
        # Save detailed results
        with open('M1/results/hyperparameter_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trial history
        trials_data = []
        for trial in self.study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            trials_data.append(trial_data)
        
        with open('M1/results/optimization_trials.json', 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        print(f"💾 Results saved to M1/results/")

def quick_hyperparameter_search():
    """Quick hyperparameter search with predefined good configurations"""
    
    print("⚡ Running Quick Hyperparameter Search")
    print("=" * 40)
    
    # Predefined configurations to test
    configs = [
        {
            'name': 'High Capacity',
            'batch_size': 8,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'use_paper_config': False,
            'enhanced_complexity': True,
            'input_size': (256, 256),
            'num_epochs': 50,
            'patience': 15
        },
        {
            'name': 'Paper Config Enhanced',
            'batch_size': 16,
            'learning_rate': 5e-4,
            'weight_decay': 1e-5,
            'use_paper_config': True,
            'enhanced_complexity': True,
            'input_size': (256, 256),
            'num_epochs': 50,
            'patience': 15
        },
        {
            'name': 'Large Input',
            'batch_size': 4,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'use_paper_config': False,
            'enhanced_complexity': True,
            'input_size': (512, 512),
            'num_epochs': 40,
            'patience': 12
        },
        {
            'name': 'Fast Training',
            'batch_size': 32,
            'learning_rate': 2e-3,
            'weight_decay': 1e-4,
            'use_paper_config': True,
            'enhanced_complexity': False,
            'input_size': (256, 256),
            'num_epochs': 30,
            'patience': 10
        }
    ]
    
    results = []
    best_config = None
    best_score = 0.0
    
    for i, config in enumerate(configs):
        print(f"\n🧪 Testing Configuration {i+1}/{len(configs)}: {config['name']}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            model, trainer, result = train_ultimate_model(config)
            training_time = time.time() - start_time
            
            config_result = {
                'config_name': config['name'],
                'config': config,
                'val_f1': result['best_val_f1'],
                'val_accuracy': result['best_val_accuracy'],
                'test_f1': result['test_f1'],
                'test_accuracy': result['test_accuracy'],
                'training_time': training_time,
                'total_epochs': result['total_epochs']
            }
            
            results.append(config_result)
            
            print(f"✅ {config['name']} completed:")
            print(f"   Val F1: {result['best_val_f1']:.4f}")
            print(f"   Test F1: {result['test_f1']:.4f}")
            print(f"   Test Accuracy: {result['test_accuracy']:.2f}%")
            print(f"   Training time: {training_time:.2f}s")
            
            if result['best_val_f1'] > best_score:
                best_score = result['best_val_f1']
                best_config = config
                
        except Exception as e:
            print(f"❌ Configuration {config['name']} failed: {str(e)}")
            continue
    
    # Save results
    with open('M1/results/quick_search_results.json', 'w') as f:
        json.dump({
            'results': results,
            'best_config': best_config,
            'best_score': best_score
        }, f, indent=2)
    
    print(f"\n🏆 Quick Search Completed!")
    print(f"🎯 Best Configuration: {best_config['name'] if best_config else 'None'}")
    print(f"🎯 Best F1 Score: {best_score:.4f}")
    
    return best_config, best_score, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for DARU-Net')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Optimization mode: quick or full')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of trials for full optimization')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds for optimization')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        best_config, best_score, results = quick_hyperparameter_search()
    else:
        optimizer = HyperparameterOptimizer(n_trials=args.trials, timeout=args.timeout)
        best_params, best_score = optimizer.optimize()
        
    print(f"\n✅ Hyperparameter optimization completed!")
    print(f"🎯 Best score achieved: {best_score:.4f}")
