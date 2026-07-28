import os
import argparse
import torch
from utils import config_loader, get_msg_mgr, init_seeds
from modeling import models


def init_opengait_logger(cfgs):
    """Initialises OpenGait's internal logging manager."""
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['evaluator_cfg']
    output_path = os.path.join(
        'output/',
        cfgs['data_cfg']['dataset_name'],
        cfgs['model_cfg']['model'],
        engine_cfg['save_name']
    )
    msg_mgr.init_logger(output_path, log_to_file=False)
    init_seeds(0)


def main():
    parser = argparse.ArgumentParser(description='Automated Adversarial Evaluation Pipeline')
    parser.add_argument('--cfgs', type=str, required=True, help="Path to config file")
    parser.add_argument('--iter', default=20000, help="Iteration checkpoint to evaluate")
    opt = parser.parse_args()

    # Set CUDA device for single-GPU execution
    torch.cuda.set_device(0)

    # 1. Load initial config just to find the starting baseline value
    base_cfgs = config_loader(opt.cfgs)
    start_val = base_cfgs.get('attack_cfg', {}).get('epsilon', 0.1)

    # 2. Safely generate a list of strengths from start_val to 1.0 in 0.1 steps
    # (Multiplying by 10 and rounding prevents weird Python floating point math like 0.300000000004)
    strengths = [round(x * 0.1, 1) for x in range(int(start_val * 10), 11)]

    modes = ['baseline', 'attack_silhouette']
    results_summary = {}
    current_strengths = [0.0] if modes == 'baseline' else strengths
    for strength in current_strengths:
        # We only need to run the baseline once, so we bypass the sweep for it


        for mode in modes:
            print("\n" + "=" * 60)
            if mode == 'baseline':
                print(f"      RUNNING EVALUATION MODE: {mode.upper()}")
            else:
                print(f"      RUNNING EVALUATION MODE: {mode.upper()} | STRENGTH: {strength}")
            print("=" * 60 + "\n")

            # Load fresh configuration for this specific evaluation pass
            cfgs = config_loader(opt.cfgs)
            cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
            cfgs['ATTACK_MODE'] = mode

            # --- OVERRIDE CONFIG HYPERPARAMETERS ---
            if 'attack_cfg' not in cfgs:
                cfgs['attack_cfg'] = {}
            cfgs['attack_cfg']['epsilon'] = strength
            cfgs['attack_cfg']['flip_prob'] = strength

            # Initialise OpenGait logger
            init_opengait_logger(cfgs)

            # Instantiate Model
            Model = getattr(models, cfgs['model_cfg']['model'])
            model = Model(cfgs, training=False).cuda()

            # Run test pass and capture accuracy dictionary
            eval_results = Model.run_test(model)

            # --- VARIABLE INJECTION ---
            eval_results['attack_strength'] = str(strength) if mode != 'baseline' else 'N/A'
            eval_results['mode_name'] = mode

            # Generate a unique dictionary key so loops don't overwrite each other (e.g. 'attack_skeleton_0.4')
            dict_key = f"{mode}_{strength}"
            results_summary[dict_key] = eval_results

    # Print Final Thesis Results Summary
    print("\n\n" + "=" * 80)
    print("         ADVERSARIAL EVALUATION SUMMARY REPORT")
    print("=" * 80)
    for key, res in results_summary.items():
        mode = res.get('mode_name', 'Unknown')
        ugs_r1 = res.get('scalar/test_accuracy/UGS@R1', 'N/A')
        fgs_r1 = res.get('scalar/test_accuracy/FGS@R1', 'N/A')
        att = res.get('attack_strength', 'N/A')

        print(f"Mode: {mode:<20} | Strength: {att:<10} | UGS@R1: {ugs_r1}% | FGS@R1: {fgs_r1}%")
    print("=" * 80)


if __name__ == '__main__':
    main()