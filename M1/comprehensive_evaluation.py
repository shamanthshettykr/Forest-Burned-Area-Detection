"""
Comprehensive Evaluation Script for DARU-Net
Provides detailed analysis of model performance with multiple metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import json
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from darunet import DARU_Net
from train_ultimate_optimized import create_advanced_datasets
from torch.utils.data import DataLoader

class ComprehensiveEvaluator:
    """Comprehensive model evaluation with detailed metrics and visualizations"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.results = {}
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        print(f"📥 Loading model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract configuration
        config = checkpoint.get('config', {})
        
        # Initialize model with same configuration
        self.model = DARU_Net(
            use_paper_config=config.get('use_paper_config', False),
            use_log_softmax=True,
            enhanced_complexity=config.get('enhanced_complexity', True)
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Validation F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
        print(f"   Validation Accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
        
    def calculate_advanced_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Advanced metrics
        balanced_accuracy = (recall + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        
        # Intersection over Union (IoU)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        # Dice coefficient
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # ROC AUC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.0
        
        # Precision-Recall AUC
        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except:
            pr_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'balanced_accuracy': balanced_accuracy,
            'mcc': mcc,
            'iou': iou,
            'dice': dice,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn), 
                'fp': int(fp), 'fn': int(fn)
            }
        }
    
    def evaluate_on_dataset(self, data_loader, dataset_name="Test"):
        """Evaluate model on a dataset"""
        print(f"\n🧪 Evaluating on {dataset_name} dataset...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (s1_imgs, s2_imgs, masks) in enumerate(tqdm(data_loader, desc=f"{dataset_name} Evaluation")):
                s1_imgs = s1_imgs.to(self.device)
                s2_imgs = s2_imgs.to(self.device)
                masks = masks.to(self.device).squeeze(1)
                
                # Forward pass
                outputs = self.model(s1_imgs, s2_imgs)
                
                # Calculate loss
                loss = F.cross_entropy(outputs, masks.long())
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy().flatten())  # Positive class prob
        
        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self.calculate_advanced_metrics(y_true, y_pred, y_prob)
        metrics['avg_loss'] = total_loss / len(data_loader)
        
        # Store results
        self.results[dataset_name.lower()] = {
            'metrics': metrics,
            'predictions': y_pred,
            'targets': y_true,
            'probabilities': y_prob
        }
        
        # Print results
        print(f"\n📊 {dataset_name} Results:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   IoU: {metrics['iou']:.4f}")
        print(f"   Dice: {metrics['dice']:.4f}")
        print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"   PR AUC: {metrics['pr_auc']:.4f}")
        print(f"   MCC: {metrics['mcc']:.4f}")
        
        return metrics
    
    def create_visualizations(self, save_dir='M1/results/evaluation_plots'):
        """Create comprehensive visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        for dataset_name, data in self.results.items():
            y_true = data['targets']
            y_pred = data['predictions']
            y_prob = data['probabilities']
            
            # 1. Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Burn', 'Burn'],
                       yticklabels=['No Burn', 'Burn'])
            plt.title(f'Confusion Matrix - {dataset_name.title()}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/confusion_matrix_{dataset_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. ROC Curve
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {dataset_name.title()}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/roc_curve_{dataset_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Precision-Recall Curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {dataset_name.title()}')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/pr_curve_{dataset_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Probability Distribution
            plt.figure(figsize=(10, 6))
            plt.hist(y_prob[y_true == 0], bins=50, alpha=0.7, label='No Burn', density=True)
            plt.hist(y_prob[y_true == 1], bins=50, alpha=0.7, label='Burn', density=True)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Density')
            plt.title(f'Probability Distribution - {dataset_name.title()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/prob_distribution_{dataset_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Visualizations saved to {save_dir}")
    
    def save_detailed_report(self, save_path='M1/results/evaluation_report.json'):
        """Save detailed evaluation report"""
        
        # Prepare report data
        report = {
            'model_path': self.model_path,
            'evaluation_timestamp': str(np.datetime64('now')),
            'device': str(self.device),
            'datasets': {}
        }
        
        # Add results for each dataset
        for dataset_name, data in self.results.items():
            report['datasets'][dataset_name] = {
                'metrics': data['metrics'],
                'sample_count': len(data['targets']),
                'positive_samples': int(np.sum(data['targets'])),
                'negative_samples': int(len(data['targets']) - np.sum(data['targets']))
            }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved to {save_path}")
        
        return report

def run_comprehensive_evaluation(model_path='M1/results/ultimate_best_model.pth'):
    """Run comprehensive evaluation on the best model"""
    
    print("🔬 Starting Comprehensive Model Evaluation")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(model_path)
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset, val_dataset, test_dataset = create_advanced_datasets(
        s1_dir='M1/data/sentinel1',
        s2_dir='M1/data/sentinel2',
        mask_dir='M1/data/masks',
        input_size=(256, 256)
    )
    
    # Create data loaders
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Evaluate on validation set
    val_metrics = evaluator.evaluate_on_dataset(val_loader, "Validation")
    
    # Evaluate on test set
    test_metrics = evaluator.evaluate_on_dataset(test_loader, "Test")
    
    # Create visualizations
    evaluator.create_visualizations()
    
    # Save detailed report
    report = evaluator.save_detailed_report()
    
    print(f"\n🎉 Comprehensive evaluation completed!")
    print(f"🎯 Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"🎯 Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"🎯 Test IoU: {test_metrics['iou']:.4f}")
    
    return evaluator, report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--model_path', type=str, default='M1/results/ultimate_best_model.pth',
                       help='Path to the trained model')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator, report = run_comprehensive_evaluation(args.model_path)
    
    print("\n✅ Evaluation completed successfully!")
