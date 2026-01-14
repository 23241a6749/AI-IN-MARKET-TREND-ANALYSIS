"""
Ablation studies to quantify contribution of each modality.
"""

import torch
import numpy as np
from typing import Dict, List
from src.training.evaluator import ModelEvaluator


class AblationStudy:
    """Performs ablation studies by removing modalities."""
    
    def __init__(self, model, device, evaluator: ModelEvaluator):
        self.model = model
        self.device = device
        self.evaluator = evaluator
    
    def remove_modality(self, dataloader, modality_to_remove: str = None):
        """
        Test model performance with a modality removed (zeroed out).
        
        Args:
            dataloader: DataLoader with test data
            modality_to_remove: 'price', 'sentiment', 'external', or None (all)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for price_seq, sentiment_seq, external_seq, targets in dataloader:
                price_seq = price_seq.to(self.device)
                sentiment_seq = sentiment_seq.to(self.device)
                external_seq = external_seq.to(self.device)
                targets = targets.to(self.device)
                
                # Zero out the specified modality
                if modality_to_remove == 'price':
                    price_seq = torch.zeros_like(price_seq)
                elif modality_to_remove == 'sentiment':
                    sentiment_seq = torch.zeros_like(sentiment_seq)
                elif modality_to_remove == 'external':
                    external_seq = torch.zeros_like(external_seq)
                elif modality_to_remove == 'all':
                    # This would test with all modalities removed (should fail)
                    pass
                
                logits, _ = self.model(price_seq, sentiment_seq, external_seq)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = self.evaluator.compute_metrics(all_targets, all_predictions)
        
        return metrics
    
    def run_full_ablation(self, dataloader) -> Dict[str, Dict]:
        """
        Run ablation study removing each modality individually.
        
        Returns:
            Dictionary with metrics for each configuration
        """
        results = {}
        
        # Full model (baseline)
        print("Evaluating full model...")
        full_metrics = self.evaluator.evaluate_model(
            self.model, dataloader, self.device, "full_model"
        )[0]
        results['full'] = full_metrics
        
        # Remove price
        print("Evaluating without price modality...")
        results['no_price'] = self.remove_modality(dataloader, 'price')
        
        # Remove sentiment
        print("Evaluating without sentiment modality...")
        results['no_sentiment'] = self.remove_modality(dataloader, 'sentiment')
        
        # Remove external
        print("Evaluating without external modality...")
        results['no_external'] = self.remove_modality(dataloader, 'external')
        
        return results
    
    def compute_contribution(self, full_metrics: Dict, ablated_metrics: Dict) -> float:
        """
        Compute the contribution of a modality as the drop in performance.
        """
        full_acc = full_metrics['accuracy']
        ablated_acc = ablated_metrics['accuracy']
        
        contribution = full_acc - ablated_acc
        
        return contribution
    
    def plot_ablation_results(self, results: Dict, save_path: str = None):
        """Plot ablation study results."""
        import matplotlib.pyplot as plt
        
        configurations = list(results.keys())
        accuracies = [results[cfg]['accuracy'] for cfg in configurations]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#3498db' if cfg == 'full' else '#e74c3c' for cfg in configurations]
        bars = ax.bar(configurations, accuracies, color=colors)
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Ablation Study: Modality Contribution')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(accuracies):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

