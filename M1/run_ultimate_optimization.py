"""
Ultimate Optimization Pipeline for DARU-Net
Orchestrates the complete optimization process from data preprocessing to final evaluation
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import subprocess

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check if data directories exist
    required_dirs = [
        'M1/data/sentinel1',
        'M1/data/sentinel2'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Required directory not found: {dir_path}")
            return False
        
        # Check if directory has images
        image_count = len(list(Path(dir_path).glob('*.png')))
        if image_count == 0:
            print(f"❌ No images found in: {dir_path}")
            return False
        
        print(f"✅ Found {image_count} images in {dir_path}")
    
    # Create results directory
    os.makedirs('M1/results', exist_ok=True)
    
    print("✅ All prerequisites met!")
    return True

def run_data_preprocessing():
    """Run data preprocessing and mask generation"""
    print("\n📊 Phase 1: Data Preprocessing and Mask Generation")
    
    command = "cd M1 && python preprocess_new_images.py"
    return run_command(command, "Data preprocessing and mask generation")

def run_quick_hyperparameter_search():
    """Run quick hyperparameter search"""
    print("\n⚡ Phase 2: Quick Hyperparameter Search")
    
    command = "cd M1 && python hyperparameter_optimization.py --mode quick"
    return run_command(command, "Quick hyperparameter search")

def run_ultimate_training(config_path=None):
    """Run ultimate training with optimized parameters"""
    print("\n🏋️ Phase 3: Ultimate Model Training")
    
    if config_path and os.path.exists(config_path):
        # Load best configuration from hyperparameter search
        with open(config_path, 'r') as f:
            search_results = json.load(f)
        
        best_config = search_results.get('best_config')
        if best_config:
            print(f"📋 Using optimized configuration: {best_config['name']}")
    
    command = "cd M1 && python train_ultimate_optimized.py"
    return run_command(command, "Ultimate model training")

def run_comprehensive_evaluation():
    """Run comprehensive model evaluation"""
    print("\n🔬 Phase 4: Comprehensive Model Evaluation")
    
    command = "cd M1 && python comprehensive_evaluation.py"
    return run_command(command, "Comprehensive model evaluation")

def run_full_hyperparameter_optimization(trials=20):
    """Run full hyperparameter optimization (optional)"""
    print(f"\n🔬 Phase 2b: Full Hyperparameter Optimization ({trials} trials)")
    
    command = f"cd M1 && python hyperparameter_optimization.py --mode full --trials {trials}"
    return run_command(command, f"Full hyperparameter optimization with {trials} trials")

def generate_final_report():
    """Generate final optimization report"""
    print("\n📄 Generating Final Report")
    
    report = {
        'optimization_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phases_completed': [],
        'results': {}
    }
    
    # Check which results files exist and load them
    result_files = {
        'preprocessing': 'M1/results/preprocessing_report.json',
        'quick_search': 'M1/results/quick_search_results.json',
        'hyperparameter_opt': 'M1/results/hyperparameter_optimization_results.json',
        'training': 'M1/results/ultimate_training_results.json',
        'evaluation': 'M1/results/evaluation_report.json'
    }
    
    for phase, file_path in result_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    report['results'][phase] = json.load(f)
                report['phases_completed'].append(phase)
                print(f"✅ Loaded {phase} results")
            except Exception as e:
                print(f"⚠️ Could not load {phase} results: {e}")
    
    # Extract key metrics
    if 'evaluation' in report['results']:
        eval_data = report['results']['evaluation']
        test_metrics = eval_data.get('datasets', {}).get('test', {}).get('metrics', {})
        
        report['final_metrics'] = {
            'test_accuracy': test_metrics.get('accuracy', 0),
            'test_f1_score': test_metrics.get('f1_score', 0),
            'test_precision': test_metrics.get('precision', 0),
            'test_recall': test_metrics.get('recall', 0),
            'test_iou': test_metrics.get('iou', 0),
            'test_dice': test_metrics.get('dice', 0),
            'test_roc_auc': test_metrics.get('roc_auc', 0)
        }
    
    # Save final report
    with open('M1/results/final_optimization_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n🎉 Final Optimization Report")
    print("=" * 50)
    print(f"📅 Completed: {report['optimization_timestamp']}")
    print(f"✅ Phases completed: {len(report['phases_completed'])}")
    
    if 'final_metrics' in report:
        metrics = report['final_metrics']
        print(f"\n🎯 Final Test Results:")
        print(f"   Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"   F1-Score: {metrics['test_f1_score']:.4f}")
        print(f"   Precision: {metrics['test_precision']:.4f}")
        print(f"   Recall: {metrics['test_recall']:.4f}")
        print(f"   IoU: {metrics['test_iou']:.4f}")
        print(f"   Dice: {metrics['test_dice']:.4f}")
        print(f"   ROC AUC: {metrics['test_roc_auc']:.4f}")
    
    print(f"\n📄 Full report saved to: M1/results/final_optimization_report.json")
    
    return report

def main():
    """Main optimization pipeline"""
    parser = argparse.ArgumentParser(description='Ultimate DARU-Net Optimization Pipeline')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing phase')
    parser.add_argument('--skip-hyperopt', action='store_true',
                       help='Skip hyperparameter optimization phase')
    parser.add_argument('--full-hyperopt', action='store_true',
                       help='Run full hyperparameter optimization instead of quick search')
    parser.add_argument('--hyperopt-trials', type=int, default=20,
                       help='Number of trials for full hyperparameter optimization')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training phase')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation phase')
    
    args = parser.parse_args()
    
    print("🚀 DARU-Net Ultimate Optimization Pipeline")
    print("=" * 60)
    print("🎯 Goal: Achieve maximum test accuracy through comprehensive optimization")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Exiting.")
        return False
    
    success_count = 0
    total_phases = 0
    
    # Phase 1: Data Preprocessing
    if not args.skip_preprocessing:
        total_phases += 1
        if run_data_preprocessing():
            success_count += 1
        else:
            print("⚠️ Data preprocessing failed, but continuing...")
    
    # Phase 2: Hyperparameter Optimization
    if not args.skip_hyperopt:
        total_phases += 1
        if args.full_hyperopt:
            if run_full_hyperparameter_optimization(args.hyperopt_trials):
                success_count += 1
        else:
            if run_quick_hyperparameter_search():
                success_count += 1
    
    # Phase 3: Ultimate Training
    if not args.skip_training:
        total_phases += 1
        config_path = 'M1/results/quick_search_results.json'
        if run_ultimate_training(config_path):
            success_count += 1
        else:
            print("❌ Training failed. Cannot proceed to evaluation.")
            return False
    
    # Phase 4: Comprehensive Evaluation
    if not args.skip_evaluation:
        total_phases += 1
        if run_comprehensive_evaluation():
            success_count += 1
    
    # Generate final report
    final_report = generate_final_report()
    
    # Summary
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n🏁 Optimization Pipeline Completed!")
    print(f"⏱️ Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"✅ Successful phases: {success_count}/{total_phases}")
    
    if success_count == total_phases:
        print("🎉 All phases completed successfully!")
        return True
    else:
        print("⚠️ Some phases failed. Check logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
