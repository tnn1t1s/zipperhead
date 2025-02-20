import argparse
from base_trainer import train_base_model
from sft_trainer import run_sft_experiment
from grpo_trainer import run_grpo_experiment
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--mode', choices=['base', 'sft', 'grpo'], required=True,
                      help='Which experiment to run')
    parser.add_argument('--base-model', help='Path to base model for SFT/GRPO',
                      required=False)
    parser.add_argument('--output-dir', default='experiments/outputs',
                      help='Directory for experiment outputs')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'base':
        base_model_path = train_base_model(args)
        print(f"Base model saved to: {base_model_path}")
    elif args.mode in ['sft', 'grpo']:
        if not args.base_model:
            raise ValueError(f"{args.mode} requires --base-model parameter")
        if args.mode == 'sft':
            run_sft_experiment(args)
        else:
            run_grpo_experiment(args)

if __name__ == "__main__":
    main()
