import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def create_base_model_plot(base_metrics_file, output_dir):
    """Figure 1: Base model convergence"""
    with open(base_metrics_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history) + 1)
    accuracies = [entry['val_accuracy'] for entry in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o')
    plt.title('Base Model Training Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.grid(True)
    plt.savefig(output_dir / 'figure1_base_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sft_plot(sft_metrics_file, output_dir):
    """Figure 2: SFT learning progression"""
    with open(sft_metrics_file, 'r') as f:
        history = json.load(f)
    
    # Extract metrics
    batches = range(1, len(history) + 1)
    odd_acc = [entry['test_odd_acc'] for entry in history]
    even_acc = [entry['test_even_acc'] for entry in history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(batches, odd_acc, marker='o', label='Odd Sums')
    plt.plot(batches, even_acc, marker='s', label='Even Sums')
    
    # Add vertical lines for batch introductions
    for batch in range(1, 11):
        plt.axvline(x=batch, color='gray', linestyle='--', alpha=0.3)
    
    plt.title('SFT Learning Progression')
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'figure2_sft_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_grpo_plot(grpo_metrics_file, output_dir):
    """Figure 3: GRPO learning dynamics"""
    with open(grpo_metrics_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history) + 1)
    odd_acc = [entry['test_odd_acc'] for entry in history]
    even_acc = [entry['test_even_acc'] for entry in history]
    rewards = [entry['reward'] for entry in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Accuracy plot
    ax1.plot(epochs, odd_acc, marker='o', label='Odd Sums')
    ax1.plot(epochs, even_acc, marker='s', label='Even Sums')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    ax1.legend()
    
    # Reward plot
    ax2.plot(epochs, rewards, color='green', marker='^')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True)
    
    plt.suptitle('GRPO Learning Dynamics')
    plt.savefig(output_dir / 'figure3_grpo_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plot(sft_metrics_file, grpo_metrics_file, output_dir):
    """Figure 4: Comparative analysis"""
    # Load metrics
    with open(sft_metrics_file, 'r') as f:
        sft_history = json.load(f)
    with open(grpo_metrics_file, 'r') as f:
        grpo_history = json.load(f)
    
    # Normalize time scales to percentage of training
    sft_progress = np.linspace(0, 100, len(sft_history))
    grpo_progress = np.linspace(0, 100, len(grpo_history))
    
    plt.figure(figsize=(12, 6))
    
    # Plot SFT
    plt.plot(sft_progress, [h['test_even_acc'] for h in sft_history], 
             label='SFT Even Sums', linestyle='-')
    
    # Plot GRPO
    plt.plot(grpo_progress, [h['test_even_acc'] for h in grpo_history], 
             label='GRPO Even Sums', linestyle='--')
    
    plt.title('Comparative Learning Trajectories')
    plt.xlabel('Training Progress (%)')
    plt.ylabel('Accuracy on Even Sums (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'figure4_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Setup directories
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths (adjust these to your actual paths)
    base_metrics = 'experiments/base/training_metrics.json'
    sft_metrics = 'experiments/sft/training_metrics.json'
    grpo_metrics = 'experiments/grpo/training_metrics.json'
    
    # Generate all figures
    create_base_model_plot(base_metrics, output_dir)
    create_sft_plot(sft_metrics, output_dir)
    create_grpo_plot(grpo_metrics, output_dir)
    create_comparison_plot(sft_metrics, grpo_metrics, output_dir)

if __name__ == '__main__':
    main()
