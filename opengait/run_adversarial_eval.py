import os
import argparse
import torch
from pyparsing import results

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
        engine_cfg['save_name'],
    )

    msg_mgr.init_logger(output_path, log_to_file=True)
    init_seeds(0)


def main():
    parser = argparse.ArgumentParser(description='Automated Adversarial Evaluation Pipeline')
    parser.add_argument('--cfgs', type=str, required=True, help="Path to config file")
    parser.add_argument('--iter', default=20000, help="Iteration checkpoint to evaluate")
    parser.add_argument('--log_to_file', action='store_true', help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
    opt = parser.parse_args()

    # Set CUDA device for single-GPU execution
    torch.cuda.set_device(0)
    cfgs = config_loader(opt.cfgs)
    modes = ['baseline', 'attack_silhouette', 'attack_skeleton', 'combined_attack']
    results_summary = {}
    #results_attack_strength = {}

    for mode in modes:
        if mode == 'attack_silhouette':
            attack_strength = cfgs['attack_cfg'].get('flip_prob')
        elif mode == 'attack_skeleton':
            attack_strength = cfgs['attack_cfg'].get('epsilon')
        elif mode == 'combined_attack':
            attack_strength = f"Epsilon: {cfgs['attack_cfg'].get('epsilon')}, Flip Prob: {cfgs['attack_cfg'].get('flip_prob')}"
        else:
            attack_strength = ''

        print("\n" + "=" * 60)
        print(f"      RUNNING EVALUATION MODE: {mode.upper()} | {attack_strength}")
        print("=" * 60 + "\n")

        # Load fresh configuration for each evaluation pass

        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['ATTACK_MODE'] = mode

        # Initialise OpenGait logger
        init_opengait_logger(cfgs)

        # Instantiate Model
        Model = getattr(models, cfgs['model_cfg']['model'])
        model = Model(cfgs, training=False).cuda()

        # Run test pass and capture accuracy dictionary
        eval_results = Model.run_test(model)
        eval_results['attack_strength'] = attack_strength
        #results_attack_strength = attack_strength
        # Store results for final summary
        results_summary[mode] = eval_results
        #results_attack_strength[mode] =


    # Print Final Thesis Results Summary
    print("\n\n" + "=" * 65)
    print("         ADVERSARIAL EVALUATION SUMMARY REPORT")
    print("=" * 65)
    for mode, res in results_summary.items():
        ugs_r1 = res.get('scalar/test_accuracy/UGS@R1')
        fgs_r1 = res.get('scalar/test_accuracy/FGS@R1')
        att = res.get('attack_strength', 'N/A')
        print(f"Mode: {mode:<20}| Strength: {att:<3} | UGS@R1: {ugs_r1}% | FGS@R1: {fgs_r1}%")
    print("=" * 65)


if __name__ == '__main__':
    main()