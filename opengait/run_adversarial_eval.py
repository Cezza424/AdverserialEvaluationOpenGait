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

    modes = ['baseline', 'attack_silhouette', 'attack_skeleton', 'combined_attack']
    results_summary = {}

    for mode in modes:
        print("\n" + "=" * 60)
        print(f"      RUNNING EVALUATION MODE: {mode.upper()}")
        print("=" * 60 + "\n")

        # Load fresh configuration for each evaluation pass
        cfgs = config_loader(opt.cfgs)
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['ATTACK_MODE'] = mode

        # Initialise OpenGait logger
        init_opengait_logger(cfgs)

        # Instantiate Model
        Model = getattr(models, cfgs['model_cfg']['model'])
        model = Model(cfgs, training=False).cuda()

        # Run test pass and capture accuracy dictionary
        eval_results = Model.run_test(model)

        # Store results for final summary
        results_summary[mode] = eval_results

    # Print Final Thesis Results Summary
    print("\n\n" + "=" * 65)
    print("         ADVERSARIAL EVALUATION SUMMARY REPORT")
    print("=" * 65)
    for mode, res in results_summary.items():
        ugs_r1 = res.get('scalar/test_accuracy/UGS@R1', 'N/A')
        fgs_r1 = res.get('scalar/test_accuracy/FGS@R1', 'N/A')
        print(f"Mode: {mode:<20} | UGS@R1: {ugs_r1}% | FGS@R1: {fgs_r1}%")
    print("=" * 65)


if __name__ == '__main__':
    main()