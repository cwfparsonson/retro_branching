from retro_branching.environments import EcoleBranching
from retro_branching.agents import StrongBranchingAgent

import ecole
import torch

import time
import numpy as np
import json
import os
from scipy import stats
import math


def extract_state_tensors_from_ecole_obs(obs, device):
    return (torch.from_numpy(obs.row_features.astype(np.float32)).to(device), 
            torch.LongTensor(obs.edge_features.indices.astype(np.int16)).to(device),
            torch.from_numpy(obs.column_features.astype(np.float32)).to(device))

def calc_num_samples_needed_for_confidence_interval(data, z_value=1.96, population_proportion=0.5, margin_of_error=0.05):
    return math.ceil( (z_value**2 * (population_proportion * (1 - population_proportion))) / (margin_of_error**2) )


if __name__ == '__main__':
    num_warmup_solves = 20 # 20
    num_solves = 100
    seed = 5
    nrows = 500 
    ncols = 1000 
    path_to_save = os.getcwd()

    env = EcoleBranching(observation_function='default',
                         information_function='default',
                         reward_function='default',
                         scip_params='default')

    ecole.seed(seed)
    instance = next(ecole.instance.SetCoverGenerator(n_rows=nrows, n_cols=ncols, density=0.05))
    instance_before_reset = instance.copy_orig()

    # init calibration agent
    calibration_agent = StrongBranchingAgent()

    # do num_warmup_solves 
    for solve in range(num_warmup_solves):
        # reset calibration env
        calibration_agent.before_reset(instance_before_reset.copy_orig())
        env.seed(seed)
        obs, action_set, reward, done, info = env.reset(instance_before_reset.copy_orig())

        while not done:
            action, _ = calibration_agent.action_select(action_set=action_set, obs=obs, munchausen_tau=0, epsilon=0, model=env.model, done=done, agent_idx=0)
            obs, action_set, reward, done, info = env.step(action)
        print(f'Completed {solve+1} of {num_warmup_solves} warm-up solves')
    print(f'Completed all {num_warmup_solves} warm-up solves.\n')

    # do num_solves 
    inference_times, step_times, total_solve_times = [], [], []
    for solve in range(num_solves):
        inference_start_times, inference_end_times = [], []
        step_start_times, step_end_times = [], []

        # reset calibration env
        calibration_agent.before_reset(instance_before_reset.copy_orig())
        env.seed(seed)
        obs, action_set, reward, done, info = env.reset(instance_before_reset.copy_orig())

        solve_start_t = time.time_ns()
        while not done:
            inference_start_times.append(time.time_ns())
            action, _ = calibration_agent.action_select(action_set=action_set, obs=obs, munchausen_tau=0, epsilon=0, model=env.model, done=done, agent_idx=0)
            inference_end_times.append(time.time_ns())

            step_start_times.append(time.time_ns())
            obs, action_set, reward, done, info = env.step(action)
            step_end_times.append(time.time_ns())
        solve_end_t = time.time_ns()

        # calc times
        inference_times.extend(((np.array(inference_end_times) - np.array(inference_start_times)) * 1e-9).tolist())
        step_times.extend(((np.array(step_end_times) - np.array(step_start_times)) * 1e-9).tolist())
        total_solve_times.append((solve_end_t - solve_start_t) * 1e-9)

        if len(total_solve_times) > 2:
            print(f'Completed {solve+1} of {num_solves} calibration solves --> curr mean inference time: {np.mean(inference_times):.3f} s, N: {calc_num_samples_needed_for_confidence_interval(inference_times):.3f} | mean step time: {np.mean(step_times):.3f} s, N: {calc_num_samples_needed_for_confidence_interval(step_times):.3f} | mean total solve time: {np.mean(total_solve_times):.3f} s, N: {calc_num_samples_needed_for_confidence_interval(total_solve_times):.3f}')
    print(f'Completed all {num_solves} calibration solves.\n')

    # create and save calibration config
    calibration_config = {'seed': seed,
                          'device': 'cpu',
                          'nrows': nrows,
                          'ncols': ncols,
                          'num_solves': num_solves,
                          'mean_inference_time': np.mean(inference_times),
                          'mean_step_time': np.mean(step_times),
                          'mean_total_solve_time': np.mean(total_solve_times),
                          'inference_times': inference_times,
                          'step_times': step_times,
                          'total_solve_times': total_solve_times}
    with open(path_to_save+'/cpu_calibration_config.json', 'w') as f:
        json.dump(calibration_config, f)
    print(f'Saved cpu_calibration_config.json to {path_to_save}/')
