# disable annoying tensorboard numpy warning
import warnings
warnings.filterwarnings(action='ignore',
                        category=FutureWarning,
                        module='tensorboard')  # noqa

from retro_branching.environments import EcoleBranching
from retro_branching.agents import PseudocostBranchingAgent, StrongBranchingAgent, Agent
from retro_branching.utils import turn_off_scip_heuristics

import ecole
import torch
from collections import defaultdict, deque
import numpy as np
import time
import pathlib
import os
import shutil
import gzip
import pickle
import json
import threading
from scipy import stats





def calibrate_hardware(calibration_config, device, kwargs):
    '''
    For CPU calibration config, records inference, step, and solve times. For GPU calibration
    config, records inference times.
    '''
    if calibration_config['device'] not in device:
        raise Exception(f'calibration_config device {calibration_config["device"]} and passed device arg {device} do not match.')

    if device == 'cpu':
        # init calibration agent
        calibration_agent = StrongBranchingAgent()

        # init calibration env
        calibration_env = EcoleBranching(observation_function='default',
                                         information_function='default',
                                         reward_function='default',
                                         scip_params='default')

        # init calibration instance
        calibration_generator = ecole.instance.SetCoverGenerator(n_rows=calibration_config['nrows'], 
                                                                 n_cols=calibration_config['ncols'], 
                                                                 density=0.05)
        calibration_generator.seed(calibration_config['seed'])
        calibration_instance = next(calibration_generator)

        for _ in range(kwargs['num_cpu_samples']):
            # reset calibration env
            calibration_env.seed(calibration_config['seed'])
            obs, action_set, reward, done, info = calibration_env.reset(calibration_instance.copy_orig())
            
            # solve calibration instance and save inference, step, and solve times
            inference_start_times, inference_end_times = [], []
            step_start_times, step_end_times = [], []
            solve_start_t = time.time_ns()
            while not done:
                inference_start_times.append(time.time_ns())
                action, _ = calibration_agent.action_select(action_set=action_set, obs=obs, munchausen_tau=0, epsilon=0, model=calibration_env.model, done=done, agent_idx=0)
                inference_end_times.append(time.time_ns())

                step_start_times.append(time.time_ns())
                obs, action_set, reward, done, info = calibration_env.step(action)
                step_end_times.append(time.time_ns())
            solve_end_t = time.time_ns()

            # calc times
            kwargs['cpu_inference_times'].extend(((np.array(inference_end_times) - np.array(inference_start_times)) * 1e-9).tolist())
            kwargs['cpu_step_times'].extend(((np.array(step_end_times) - np.array(step_start_times)) * 1e-9).tolist())
            kwargs['cpu_total_solve_times'].append((solve_end_t - solve_start_t) * 1e-9)

    elif 'cuda' in device:
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

        if 'obs' not in kwargs or 'action_set' not in kwargs:
            # init env and instance
            calibration_env = EcoleBranching(observation_function='default',
                                             information_function='default',
                                             reward_function='default',
                                             scip_params='default')
            calibration_generator = ecole.instance.SetCoverGenerator(n_rows=calibration_config['nrows'], 
                                                                     n_cols=calibration_config['ncols'], 
                                                                     density=0.05)
            calibration_generator.seed(calibration_config['seed'])

            calibration_instance = next(calibration_generator)

            # reset calibration env
            calibration_env.seed(calibration_config['seed'])
            calibration_agent.before_reset(calibration_instance)
            obs, action_set, reward, done, info = calibration_env.reset(calibration_instance.copy_orig())

            # do any conversions
            action_set = action_set.astype(int) # ensure action set is int so gets correctly converted to torch.LongTensor later
            obs = extract_state_tensors_from_ecole_obs(obs, device) 
        else:
            action_set, obs = kwargs['action_set'], kwargs['obs']

        # get inference times
        for _ in range(kwargs['num_gpu_samples']):
            inference_start_t = torch.cuda.Event(enable_timing=True)
            inference_end_t = torch.cuda.Event(enable_timing=True)
            inference_start_t.record()
            action, _ = calibration_agent.action_select(action_set=action_set, obs=obs)
            inference_end_t.record()
            torch.cuda.synchronize(device=device)
            inference_time = (inference_start_t.elapsed_time(inference_end_t)) * 1e-3
            kwargs['gpu_inference_times'].append(inference_time)

    else:
        raise Exception(f'Unrecognised device {device}')


# def extract_state_tensors_from_ecole_obs(obs, device):
    # return (torch.from_numpy(obs.row_features.astype(np.float32)).to(device), 
            # torch.LongTensor(obs.edge_features.indices.astype(np.int16)).to(device),
            # torch.from_numpy(obs.variable_features.astype(np.float32)).to(device))
def extract_state_tensors_from_ecole_obs(obs, device):
    return (torch.from_numpy(obs.row_features.astype(np.float32)).to(device), 
            torch.LongTensor(obs.edge_features.indices.astype(np.int16)).to(device),
            torch.from_numpy(obs.edge_features.values.astype(np.float32)).to(device).unsqueeze(1),
            torch.from_numpy(obs.variable_features.astype(np.float32)).to(device))




class ReinforcementLearningValidator:
    def __init__(self,
                 agents,
                 envs,
                 instances,
                 calibration_config_path=None,
                 calibration_freq=10,
                 num_cpu_calibrations=15,
                 num_gpu_calibrations=500,
                 metrics=['num_nodes'],
                 seed=0,
                 max_steps=int(1e12),
                 max_steps_agent=None,
                 turn_off_heuristics=False,
                 max_threshold_difficulty=None,
                 min_threshold_difficulty=None,
                 episode_log_frequency=1,
                 checkpoint_frequency=1,
                 path_to_save=None,
                 overwrite=False,
                 name='rl_validator',
                 **kwargs):
        '''
        agents keys must be agent names/ids and envs keys must be same as agents keys
        (i.e. each agent has its own env).
        
        Args:
            agents (dict)
            envs (dict)
            instances (generator)
            calibration_config_path (None, str): If not None, will calculate and record
                calibration solve time similar to in https://arxiv.org/pdf/2012.13349.pdf.
                Path should point towards a directory contianing i) cpu_calibration_config.json
                and ii) gpu_calibration_config.json
            metrics (list): List of reward metrics to track.
            max_steps (int): Maximum number of steps in episode to use RL agent for.
            max_steps_agent (obj, None): Agent to use for remaining steps in epsiode after max_steps.
                If None, will terminate episode after max_steps.

        kwargs:
            threshold_agent (object)
            threshold_env (object)

        '''
        self.agents = agents
        self.envs = envs
        self.instances = instances
        self.calibration_config_path = calibration_config_path
        self.calibration_freq = calibration_freq
        self.num_cpu_calibrations = num_cpu_calibrations
        self.num_gpu_calibrations = num_gpu_calibrations
        self.calibration_obs, self.calibration_action_set = None, None
        if self.calibration_config_path is not None:
            with open(self.calibration_config_path+'/cpu_calibration_config.json') as f:
                self.cpu_calibration_config = json.load(f)
            with open(self.calibration_config_path+'/gpu_calibration_config.json') as f:
                self.gpu_calibration_config = json.load(f)
            self.reset_calibation_time_arrays()
        self.metrics = metrics
        self.seed = seed
        self.max_steps = max_steps
        self.max_steps_agent = max_steps_agent
        self.turn_off_heuristics = turn_off_heuristics
        
        self.min_threshold_difficulty = min_threshold_difficulty
        self.max_threshold_difficulty = max_threshold_difficulty
        if self.min_threshold_difficulty is not None or self.max_threshold_difficulty is not None:
            # init threshold env for evaluating difficulty when generating instance
            if 'threshold_env' not in kwargs:
                self.threshold_env = EcoleBranching(observation_function=list(envs.values())[0].str_observation_function,
                                                   information_function=list(envs.values())[0].str_information_function,
                                                   reward_function=list(envs.values())[0].str_reward_function,
                                                   scip_params=list(envs.values())[0].str_scip_params)
                # self.threshold_env = EcoleBranching()
            else:
                self.threshold_env = kwargs['threshold_env']
            if 'threshold_agent' not in kwargs:
                self.threshold_agent = PseudocostBranchingAgent()
            else:
                self.threshold_agent = kwargs['threshold_agent']
            self.threshold_env.seed(self.seed)
            self.threshold_agent.eval() # put in evaluation mode
        else:
            self.threshold_env = None
            self.threshold_agent = None
        self.episode_log_frequency = episode_log_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_counter = 1
        self.path_to_save = path_to_save
        self.overwrite = overwrite
        self.name = name

        # init directory to save data
        if self.path_to_save is not None:
            self.path_to_save = self.init_save_dir(path=self.path_to_save)
        
        # ensure all envs have same seed for consistent resetting of instances
        for env in self.envs.values():
            env.seed(self.seed)
        
        self.curr_instance = None
        
        self.episodes_log = self.init_episodes_log()

        self.kwargs = kwargs

        try:
            self.device = self.agents[list(self.agents.keys())[0]].device
        except AttributeError:
            # agent does not have device parameter, assume is e.g. strong branching and is on CPU
            self.device = 'cpu'

    def reset_calibation_time_arrays(self):
        self.cpu_calibration_inference_times = []
        self.cpu_calibration_step_times = []
        self.cpu_calibration_total_solve_times = []
        self.gpu_calibration_inference_times = []

    def init_save_dir(self, path='.'):
        
        # init folder to save data 
        _path = path + '/{}/'.format(self.name)
        pathlib.Path(_path).mkdir(parents=True, exist_ok=True)
        counter = 1
        foldername = '{}_{}/'
        if not self.overwrite:
            while os.path.isdir(_path+foldername.format(self.name, counter)):
                counter += 1
        else:
            if os.path.isdir(_path+foldername.format(self.name, counter)):
                shutil.rmtree(_path+foldername.format(self.name, counter))
        foldername = foldername.format(self.name, counter)
        os.mkdir(_path+foldername)

        return _path+foldername

    def save_checkpoint(self):
        # make checkpoint dir
        path = self.path_to_save + 'checkpoint_{}/'.format(self.checkpoint_counter)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # save checkpoint
        self.save(path=path)
        print('Saved checkpoint to {}'.format(path))
        self.checkpoint_counter += 1

    def save(self, log_name='episodes_log', path='.'):
        # save episodes log
        filename = path+'{}.pkl'.format(log_name)
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self.episodes_log, f)
            
    def reset_env(self, env, max_attempts=10000):

        counter = 1
        while True:
            instance = next(self.instances)
            if self.turn_off_heuristics:
                instance = turn_off_scip_heuristics(instance)
            instance_before_reset = instance.copy_orig()
            env.seed(self.seed)
            obs, action_set, reward, done, info = env.reset(instance)
            # print(f'obs:{obs}\nreward: {reward}\ndone:{done}\ninfo:{info}')

            if not done:
                if self.min_threshold_difficulty is not None or self.max_threshold_difficulty is not None:
                    # check difficulty using threshold agent
                    meets_threshold = False
                    self.threshold_env.seed(self.seed)
                    _obs, _action_set, _reward, _done, _info = self.threshold_env.reset(instance_before_reset.copy_orig())
                    while not _done:
                        _action, _action_idx = self.threshold_agent.action_select(action_set=_action_set, obs=_obs, agent_idx=0)
                        # _action = _action_set[_action]
                        _obs, _action_set, _reward, _done, _info = self.threshold_env.step(_action)
                        if self.max_threshold_difficulty is not None:
                            if _info['num_nodes'] > self.max_threshold_difficulty:
                                # already exceeded threshold difficulty
                                break
                    if self.min_threshold_difficulty is not None:
                        if _info['num_nodes'] >= self.min_threshold_difficulty:
                            meets_threshold = True
                    if self.max_threshold_difficulty is not None:
                        if _info['num_nodes'] <= self.max_threshold_difficulty:
                            meets_threshold = True
                        else:
                            meets_threshold = False
                    if meets_threshold:
                        # can give instance to agent to learn on
                        self.curr_instance = instance_before_reset.copy_orig()
                        return obs, action_set, reward, done, info, instance_before_reset
                else:
                    self.curr_instance = instance_before_reset.copy_orig()
                    return obs, action_set, reward, done, info, instance_before_reset

            counter += 1
            if counter > max_attempts:
                raise Exception('Unable to generate valid instance after {} attempts.'.format(max_attempts))
                
    def init_episodes_log(self):
        episodes_log = {}
        for agent in self.agents.keys():
            episodes_log[agent] = defaultdict(list)
        episodes_log['metrics'] = self.metrics
        episodes_log['turn_off_heuristics'] = self.turn_off_heuristics
        episodes_log['min_threshold_difficulty'] = self.min_threshold_difficulty
        episodes_log['max_threshold_difficulty'] = self.max_threshold_difficulty
        episodes_log['agent_names'] = list(self.agents.keys())
                
        return episodes_log

    # def check_95_confidence(self, list_of_data):
        # '''list_of_data should be list of lists.'''
        # for data in list_of_data:
            # if stats.sem(data) > 1.96:
                # return False
        # return True


    def calibrate_cpu(self):
        kwargs = {'cpu_inference_times': self.cpu_calibration_inference_times,
                  'cpu_step_times': self.cpu_calibration_step_times,
                  'cpu_total_solve_times': self.cpu_calibration_total_solve_times,
                  'num_cpu_samples': self.num_cpu_calibrations}
        thread = threading.Thread(target=calibrate_hardware, 
                                  args=(self.cpu_calibration_config, 
                                        'cpu',
                                        kwargs,))
        thread.start()
        return thread

    def calibrate_gpu(self, perform_warmup=True):
        if self.calibration_obs is None or self.calibration_action_set is None:
            # do resetting once outside function so do not have to needlessly repeat
            calibration_env = EcoleBranching(observation_function='default',
                                             information_function='default',
                                             reward_function='default',
                                             scip_params='default')
            calibration_generator = ecole.instance.SetCoverGenerator(n_rows=self.gpu_calibration_config['nrows'], 
                                                                     n_cols=self.gpu_calibration_config['ncols'], 
                                                                     density=0.05)
            calibration_generator.seed(self.gpu_calibration_config['seed'])

            calibration_instance = next(calibration_generator)

            # reset calibration env
            calibration_env.seed(self.gpu_calibration_config['seed'])
            obs, action_set, reward, done, info = calibration_env.reset(calibration_instance.copy_orig())

            # do any conversions
            self.calibration_action_set = action_set.astype(int) # ensure action set is int so gets correctly converted to torch.LongTensor later
            self.calibration_obs = extract_state_tensors_from_ecole_obs(obs, self.device) 

        if perform_warmup:
            # first episode, warm-up GPUs without saving calibrations
            kwargs = {'gpu_inference_times': [],
                      'obs': self.calibration_obs,
                      'action_set': self.calibration_action_set,
                      'num_gpu_samples': self.num_gpu_calibrations}
            calibrate_hardware(self.gpu_calibration_config, 
                                        self.device, 
                                        kwargs)
        # run gpu calibration on gpu
        kwargs = {'gpu_inference_times': self.gpu_calibration_inference_times,
                  'obs': self.calibration_obs,
                  'action_set': self.calibration_action_set,
                  'num_gpu_samples': self.num_gpu_calibrations}
        calibrate_hardware(self.gpu_calibration_config, 
                                   self.device, 
                                   kwargs)


    def calibrate(self, episode, debug_mode=False, verbose=True):
        calibration_start_t = time.time()

        # init time measurement arrays
        self.reset_calibation_time_arrays()

        # run cpu calibration on thread
        thread = self.calibrate_cpu()

        if 'cuda' in self.device:
            # run gpu calibration
            self.calibrate_gpu(perform_warmup=(episode == 0))

        # need cpu thread to be completed so can calc calibrated times
        thread.join()

        if verbose:
            verbose_str = f'Completed hardware calibration in {(time.time() - calibration_start_t):.3f} s || CPU inference | step | solve: {np.mean(self.cpu_calibration_inference_times):.3f} s | {np.mean(self.cpu_calibration_step_times):.3f} s | {np.mean(self.cpu_calibration_total_solve_times):.3f} s'
            if 'cuda' in self.device:
                verbose_str += f' || GPU inference: {np.mean(self.gpu_calibration_inference_times):.3f} s'
            print(verbose_str)
        if debug_mode:
            print(f'cpu_inference_times: {self.cpu_calibration_inference_times}')
            print(f'cpu_step_times: {self.cpu_calibration_step_times}')
            print(f'cpu_calibration_total_solve_times: {self.cpu_calibration_total_solve_times}')
            print(f'gpu_inference_times: {self.gpu_calibration_inference_times}')


        
    def test(self, num_episodes):
        self.episode_counter = 0
        for episode in range(num_episodes):
            if self.calibration_config_path is not None:
                if self.episode_counter % self.calibration_freq == 0:
                    # perform calibrations
                    self.calibrate(episode)

            # solve instance with agent
            self.run_episode(episode)

        # save validation data
        self.save_checkpoint()

    
    def run_episode(self, episode):
        # use first env to find instance which is not pre-solve (returns None) on env.reset()
        env = self.envs[list(self.envs.keys())[0]]
        try:
            _, _, _, _, _, instance_before_reset = self.reset_env(env=env)
        except StopIteration:
            # ran out of iteratons, cannot run any more episodes
            print('Ran out of iterations.')
            return
        
        # print('\n\n>>> NEW INSTANCE <<<')
        for agent_name in self.agents.keys():
            # print('\nAgent name: {}'.format(agent_name))
            env = self.envs[agent_name]
            agent = self.agents[agent_name]
            start = time.time()
            with torch.no_grad():
                episode_stats = defaultdict(list)
                env.seed(self.seed)
                obs, action_set, reward, done, info = env.reset(instance_before_reset.copy_orig())
                if action_set is not None:
                    action_set = action_set.astype(int) # ensure action set is int so gets correctly converted to torch.LongTensor later

                for t in range(self.max_steps):
                    if 'cuda' in self.device:
                        # extract input for DNN
                        obs = extract_state_tensors_from_ecole_obs(obs, self.device) 
                    
                    # solve for this step
                    solve_start_t = time.time_ns()
                    if agent.name != 'scip_branching':
                        # get agent action
                        if 'cuda' in self.device:
                            inference_start_t = torch.cuda.Event(enable_timing=True)
                            inference_end_t = torch.cuda.Event(enable_timing=True)
                            inference_start_t.record()
                        else:
                            inference_start_t = time.time_ns()
                        action, action_idx = agent.action_select(action_set=action_set, obs=obs, munchausen_tau=0, epsilon=0, model=env.model, done=done, agent_idx=0)
                        if 'cuda' in self.device:
                            torch.cuda.synchronize(device=self.device)
                            inference_end_t.record()
                        else:
                            inference_end_t = time.time_ns()
                        try:
                            episode_stats['action_probabilities'].append(np.array(agent.probs.cpu()))
                        except AttributeError:
                            # agent has no action probabilities
                            pass
                    else:
                        # using default scip branching heuristic in configuring env
                        action = {}

                    # take step in environments
                    step_start_t = time.time_ns()
                    obs, action_set, reward, done, info = env.step(action)
                    step_end_t = time.time_ns()

                    # calc times
                    if agent.name != 'scip_branching':
                        if 'cuda' in self.device:
                            inference_time = (inference_start_t.elapsed_time(inference_end_t)) * 1e-3
                        else:
                            inference_time = max((inference_end_t - inference_start_t)*1e-9, 1e-12) # ensure not 0
                        step_time = max((step_end_t - step_start_t)*1e-9, 1e-12) # ensure not 0
                        solve_time = inference_time + step_time
                    else:
                        solve_time = inference_time = step_time = max((step_end_t - solve_start_t)*1e-9, 1e-12)

                    if self.calibration_config_path is not None:
                        if agent.name != 'scip_branching':
                            # calc calibrated inference time
                            if 'cuda' in self.device:
                                elapsed_calibrated_inferences = (inference_time) / np.mean(self.gpu_calibration_inference_times)
                                elapsed_calibrated_inference_time = elapsed_calibrated_inferences * self.gpu_calibration_config['mean_inference_time']
                            else:
                                elapsed_calibrated_inferences = (inference_time) / np.mean(self.cpu_calibration_inference_times)
                                elapsed_calibrated_inference_time = elapsed_calibrated_inferences * self.cpu_calibration_config['mean_inference_time']
                            
                            # calc calibrated step time
                            elapsed_calibrated_steps = (step_time) / np.mean(self.cpu_calibration_step_times)
                            elapsed_calibrated_step_time = elapsed_calibrated_steps * self.cpu_calibration_config['mean_step_time']

                            # calc total calibrated solve time
                            elapsed_calibrated_solves = elapsed_calibrated_inferences + elapsed_calibrated_steps
                            elapsed_calibrated_solve_time = elapsed_calibrated_inference_time + elapsed_calibrated_step_time
                        else:
                            elapsed_calibrated_solves = (solve_time) / np.mean(self.cpu_calibration_total_solve_times)
                            elapsed_calibrated_solve_time = elapsed_calibrated_solves * self.cpu_calibration_config['mean_total_solve_time']
                            elapsed_calibrated_steps = elapsed_calibrated_inferences = elapsed_calibrated_solves
                            elapsed_calibrated_step_time = elapsed_calibrated_inference_time = elapsed_calibrated_solve_time

                        if np.isnan(elapsed_calibrated_solve_time) or np.isnan(elapsed_calibrated_inference_time) or np.isnan(elapsed_calibrated_step_time):
                            raise Exception('NaN time value found')

                    if action_set is not None:
                        action_set = action_set.astype(int) # ensure action set is int so gets correctly converted to torch.LongTensor later

                    # record metrics
                    if agent.name == 'scip_branching':
                        # no reward in configuring env, set as info
                        reward = info
                    for metric in self.metrics:
                        episode_stats[metric].append(reward[metric])
                    # episode_stats['elapsed_solve_time'].append(elapsed_solve_time)
                    episode_stats['solve_time'].append(solve_time)
                    if agent.name != 'scip_branching':
                        episode_stats['step_time'].append(step_time)
                        episode_stats['inference_time'].append(inference_time)
                    if self.calibration_config_path is not None:
                        episode_stats['elapsed_calibrated_solves'].append(elapsed_calibrated_solves)
                        episode_stats['elapsed_calibrated_solve_time'].append(elapsed_calibrated_solve_time)
                        episode_stats['elapsed_calibrated_inferences'].append(elapsed_calibrated_inferences)
                        episode_stats['elapsed_calibrated_inference_time'].append(elapsed_calibrated_inference_time)
                        episode_stats['elapsed_calibrated_steps'].append(elapsed_calibrated_steps)
                        episode_stats['elapsed_calibrated_step_time'].append(elapsed_calibrated_step_time)
                    m = env.model.as_pyscipopt()
                    episode_stats['dual_bound'].append(m.getDualbound())
                    episode_stats['primal_bound'].append(m.getPrimalbound())
                    episode_stats['gap'].append(m.getGap())

                    if done:
                        break

                if not done:
                    # reached max_steps
                    print('Reached max steps at t={}. Num nodes: {}'.format(t, info['num_nodes']))
                    if self.max_steps_agent is not None:
                        # use max_steps agent to solve rest of episode
                        while not done:
                            # get action
                            action, action_idx = self.max_steps_agent.action_select(action_set=action_set, obs=obs, agent_idx=0)
                            # action = action_set[action.item()]
                            try:
                                episode_stats['action_probabilities'].append(np.array(self.max_steps_agent.probs.cpu()))
                            except AttributeError:
                                # agent has no action probabilities
                                pass
                            # take step in environments
                            obs, action_set, reward, done, info = env.step(action)
                            if action_set is not None:
                                action_set = action_set.astype(int) # ensure action set is int so gets correctly converted to torch.LongTensor later
                            for metric in self.metrics:
                                episode_stats[metric].append(reward[metric])

            # finished test episode for this agent
            end = time.time()
            episode_stats['episode_run_time'] = end-start
            self.update_episodes_log(agent_name, episode_stats)
            
        # finished episode for all agents
        if self.episode_counter % self.episode_log_frequency == 0 and self.episode_log_frequency != float('inf'):
            print(self.get_episode_log_str())
        if self.path_to_save is not None:
            if self.episode_counter % self.checkpoint_frequency == 0:
                self.save_checkpoint()
        self.episode_counter += 1
        
    def update_episodes_log(self, agent, episode_stats):
        for key, val in episode_stats.items():
            self.episodes_log[agent][key].append(val)


    def get_episode_log_str(self):
        '''Returns string logging end of episode.'''
        log_str = 'Episode {}'.format(self.episode_counter)
        for agent in self.agents.keys():
            log_str += ' || {}: Run time: {} s'.format(agent, round(self.episodes_log[agent]['episode_run_time'][-1], 3))
            log_str += ' | Nodes: {}'.format(abs(np.sum(self.episodes_log[agent]['num_nodes'][-1])))
            log_str += ' | LP iters: {}'.format(abs(np.sum(self.episodes_log[agent]['lp_iterations'][-1])))
            if self.calibration_config_path is not None:
                log_str += ' | Cal. solve time: {} s'.format(round(abs(np.sum(self.episodes_log[agent]['elapsed_calibrated_solve_time'][-1])), 3))
                # log_str += ' | Solve time: {} s'.format(round(abs(np.sum(self.episodes_log[agent]['elapsed_solve_time'][-1])), 3))
            else:
                log_str += ' | Solve time: {} s'.format(round(abs(np.sum(self.episodes_log[agent]['solving_time'][-1])), 3))
        return log_str



class SupervisedLearningValidator:
    def __init__(self):
        pass
            
