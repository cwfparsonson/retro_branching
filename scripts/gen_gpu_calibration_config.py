from retro_branching.environments import EcoleBranching
from retro_branching.agents import Agent

import ecole
import torch

import time
import numpy as np
import json
import os
from scipy import stats


def extract_state_tensors_from_ecole_obs(obs, device):
    return (torch.from_numpy(obs.row_features.astype(np.float32)).to(device), 
            torch.LongTensor(obs.edge_features.indices.astype(np.int16)).to(device),
            torch.from_numpy(obs.column_features.astype(np.float32)).to(device))


if __name__ == '__main__':
    device = 'cuda:1'
    num_warmup_inferences = 1000
    num_inferences = 10000
    seed = 5
    nrows = 7500
    ncols = 7500
    path_to_save = os.getcwd()

    env = EcoleBranching(observation_function='default',
                         information_function='default',
                         reward_function='default',
                         scip_params='default')

    ecole.seed(seed)
    instance = next(ecole.instance.SetCoverGenerator(n_rows=nrows, n_cols=ncols, density=0.05))
    instance_before_reset = instance.copy_orig()

    # init calibration agent
    path = '/scratch/datasets/retro_branching/supervised_learner/gnn/gnn_343/checkpoint_233/'
    config = path + 'config.json'
    calibration_agent = Agent(device=device, config=config)
    for network_name, network in calibration_agent.get_networks().items():
        if network_name == 'networks':
            # TEMPORARY: Fix typo
            network_name = 'network'
        if network is not None:
            try:
                # see if network saved under same var as 'network_name'
                calibration_agent.__dict__[network_name].load_state_dict(torch.load(path+f'/{network_name}_params.pkl', map_location=device))
            except KeyError:
                # network saved under generic 'network' var (as in Agent class)
                calibration_agent.__dict__['network'].load_state_dict(torch.load(path+f'/{network_name}_params.pkl', map_location=device))
        else:
            pass
    calibration_agent.eval()

    # reset calibration env
    calibration_agent.before_reset(instance_before_reset.copy_orig())
    env.seed(seed)
    obs, action_set, reward, done, info = env.reset(instance_before_reset.copy_orig())

    action_set = action_set.astype(int)
    obs = extract_state_tensors_from_ecole_obs(obs, device)

    # do num_warmup_inferences
    for inference in range(num_warmup_inferences):
        action, _ = calibration_agent.action_select(action_set=action_set, obs=obs, munchausen_tau=0, epsilon=0, model=env.model, done=done, agent_idx=0)
        print(f'Completed {inference+1} of {num_warmup_inferences} warm-up inferences')
    print(f'Completed all {num_warmup_inferences} warm-up inferences.\n')

    # do num_inferences
    inference_times = []
    for inference in range(num_inferences):
        inference_start_t = time.time_ns()
        action, _ = calibration_agent.action_select(action_set=action_set, obs=obs, munchausen_tau=0, epsilon=0, model=env.model, done=done, agent_idx=0)
        torch.cuda.synchronize(device=device)
        inference_times.append((time.time_ns() - inference_start_t)*1e-9)
        print(f'Completed {inference+1} of {num_inferences} calibration inferences --> curr mean inference time: {np.mean(inference_times):.3f} s, standard error: {stats.sem(inference_times):.3f}')
    print(f'Completed all {num_inferences} calibration inferences.\n')

    # create and save calibration config
    calibration_config = {'seed': seed,
                          'device': 'cuda',
                          'nrows': nrows,
                          'ncols': ncols,
                          'num_inferences': num_inferences,
                          'mean_inference_time': np.mean(inference_times),
                          'inference_times': inference_times}
    with open(path_to_save+'/gpu_calibration_config.json', 'w') as f:
        json.dump(calibration_config, f)
    print(f'Saved gpu_calibration_config.json to {path_to_save}/')
