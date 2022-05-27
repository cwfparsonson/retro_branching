from retro_branching.learners import Learner
from retro_branching.environments import EcoleConfiguring, EcoleBranching
from retro_branching.agents import StrongBranchingAgent, PseudocostBranchingAgent, PseudocostBranchingAgent, REINFORCEAgent
from retro_branching.utils import turn_off_scip_heuristics, pad_tensor
from retro_branching.validators import ReinforcementLearningValidator
from retro_branching.networks import BipartiteGCN
from retro_branching.loss_functions import CrossEntropy

import ecole
import pyscipopt

import torch
import torch.nn.functional as F

from collections import defaultdict
import os
import numpy as np
import time
import pathlib
import gzip
import pickle
import json
import copy
import abc




class REINFORCELearner(Learner):
    def __init__(self,
                 agent,
                 env,
                 instances,
                 seed=0,
                 max_steps=int(1e12),
                 max_steps_agent=None,
                 batch_size=None,
                 baseline=None,
                 agent_reward='num_nodes',
                 # scale_episode_rewards=False,
                 lr=1e-4,
                 gamma=0.99,
                 # max_log_probs=float('inf'),
                 turn_off_heuristics=False,
                 threshold_difficulty=None,
                 action_filter_agent=None,
                 action_filter_percentile=90,
                 validation_frequency=None,
                 episode_log_frequency=1,
                 checkpoint_frequency=1,
                 path_to_save='.',
                 name='reinforce_learner',
                 **kwargs):
        '''
        To try overfitting, provide instances arg as a single instance. N.B.
        Before providing a single instance, you should ensure that if env.reset()
        is ran enough time on the instance then it will not pre-solve (i.e. obs != None),
        otherwise learner will raise an error since it won't be able to generate
        a valid learning instance.

        Args:
            scale_episode_rewards (bool): If True, when episode is finished, will
                scale each reward in episode to sense*(0, 1) depending on how much
                reward contributed to overall reward before discounting.
            baseline (None, 'sb', 'pc', 'gr', 'sr')
            max_steps (int): Maximum number of steps in episode to use RL agent for.
            max_steps_agent (obj, None): Agent to use for remaining steps in epsiode after max_steps.
                If None, will terminate episode after max_steps.
            action_filter_agent (object): If not None, will use action_filter_agent.extract(model, done)
                to extract expert scores for each action, and will then filter out
                actions with poor scores and only present the agent with the top
                action_filter_percentile actions to choose from.
            action_filter_percentile (int): Actions which score above this percentile relative
                to other actions will be those given to the agent. Must be between
                0 and 100 inclusive. A value of 0 will filter no actions, and a value
                of 100 will only give the RL agent the action considered best by
                the action_filter_agent.

        kwargs:
            greedy_rollout_frequency (int): How often (no. episodes) to evaluate 
                agent vs. baseline_agent -> update baseline_agent if agent better.
            greedy_rollout_evaluations (int): How many instances to evaluate
                agent vs. baseline_agent on.
            sampled_rollout_beams (int): How many beams to use.
            sampled_rollout_frequency (int): How often (no. steps) to rollout beams
                to get expected discounted reward at current step.
            apply_max_steps_to_rollout (bool): If True, will use rollout beam agent
                only for first max_steps in rollout env before switching to max_steps_agent
                if max_steps_agent is not None (otherwise terminates episode).
            threshold_agent (object, 'baseline_agent'): Either a REINFORCEAgent or
                string 'baseline_agent' to set as just being whatever baseline is.
            threshold_env (object)
            
        '''
        super(REINFORCELearner, self).__init__(agent, path_to_save, name)

        self.agent = agent
        self.agent.train() # turn on training mode
        self.env = env
        self.seed = seed
        self.env.seed(self.seed)
        self.instances = instances
        if type(self.instances) == ecole.core.scip.Model:
            # have been given one en
            print('Have been given one instance, will overfit to this instance.')
            self.overfit_instance = True
            self.curr_instance = instances.copy_orig()
        else:
            # will generate different instances for training
            self.overfit_instance = False
            self.curr_instance = None

        self.max_steps = max_steps
        self.max_steps_agent = max_steps_agent
        self.batch_size = batch_size
        self.agent_reward = agent_reward
        # self.scale_episode_rewards = scale_episode_rewards 
        self.lr = lr
        self.gamma = gamma
        # self.max_log_probs = max_log_probs

        self.baseline = baseline
        if self.baseline is not None and self.baseline != 'mean':
            self.baseline_env = EcoleBranching(observation_function=self.env.str_observation_function,
                                               information_function=self.env.str_information_function,
                                               reward_function=self.env.str_reward_function,
                                               scip_params=self.env.str_scip_params)
            # self.baseline_env = EcoleBranching()
            self.baseline_env.seed(self.seed)
            if self.baseline == 'sb':
                self.baseline_agent = StrongBranchingAgent()
            elif self.baseline == 'pc':
                self.baseline_agent = PseudocostBranchingAgent()
            elif self.baseline == 'gr' or self.baseline == 'sr':
                self.baseline_agent = REINFORCEAgent(device=self.agent.device,
                                                     config=self.agent.create_config())
                self.baseline_agent.policy_network.load_state_dict(self.agent.policy_network.state_dict())
                if self.baseline_agent.filter_network is not None:
                    self.baseline_agent.load_state_dict(self.agent.filter_network.state_dict())
                self.baseline_agent.eval() # put in evaluation mode
                self.prev_beam_baseline = None # init previous beam baseline tracker
            else:
                raise Exception('Unrecognised baseline {}'.format(self.baseline))
        else:
            self.baseline_env = None

        self.turn_off_heuristics = turn_off_heuristics
        self.threshold_difficulty = threshold_difficulty
        if self.threshold_difficulty is not None:
            # init threshold env for evaluating difficulty when generating instance
            if 'threshold_env' not in kwargs:
                self.threshold_env = EcoleBranching(observation_function=self.env.str_observation_function,
                                                    information_function=self.env.str_information_function,
                                                    reward_function=self.env.str_reward_function,
                                                    scip_params=self.env.str_scip_params)
                # self.threshold_env = EcoleBranching()
            else:
                self.threshold_env = kwargs['threshold_env']
            if 'threshold_agent' not in kwargs:
                self.threshold_agent = PseudocostBranchingAgent()
            else:
                # user has provided threshold agent
                if kwargs['threshold_agent'] == 'baseline_agent':
                    # threshold agent is baseline agent
                    self.threshold_agent = self.baseline_agent
                else:
                    # threshold agent provided by user
                    self.threshold_agent = kwargs['threshold_agent']
            self.threshold_env.seed(self.seed)
            self.threshold_agent.eval() # put in evaluation mode
        else:
            self.threshold_env = None
            self.threshold_agent = None

        self.action_filter_agent = action_filter_agent
        self.action_filter_percentile = action_filter_percentile

        self.validation_frequency = validation_frequency
        self.optimizer = self.reset_optimizer(lr=self.lr)
        self.episode_log_frequency = episode_log_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.path_to_save = path_to_save

        self.name = name
        self.eps = np.finfo(np.float32).eps.item()

        # init save directory
        self.path_to_save = self.init_save_dir(path=self.path_to_save)

        self.episodes_log = self.init_episodes_log()

        self.kwargs = self.init_kwargs(kwargs)

        if self.overfit_instance and self.baseline in ['gr', 'sr']:
            raise Exception('If using rollout baseline, instances must be an instances generator i.e. cannot try to overfit.')


    def reset_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr)
        return self.optimizer

    def init_kwargs(self, kwargs):
        if 'greedy_rollout_frequency' not in kwargs:
            kwargs['greedy_rollout_frequency'] = 10
        if 'greedy_rollout_evaluations' not in kwargs:
            kwargs['greedy_rollout_evaluations'] = 10
        if 'sampled_rollout_beams' not in kwargs:
            kwargs['sampled_rollout_beams'] = 1
        if 'sampled_rollout_frequency' not in kwargs:
            kwargs['sampled_rollout_frequency'] = 2
        if 'apply_max_steps_to_rollout' not in kwargs:
            kwargs['apply_max_steps_to_rollout'] = True
        
        return kwargs

    def reset_env(self, env, max_attempts=10000):

        # obs, counter = None, 1
        # while obs is None:
        counter = 1
        while True:
            if self.overfit_instance:
                instance = self.curr_instance.copy_orig()
            else:
                instance = next(self.instances)
            if self.turn_off_heuristics:
                instance = turn_off_scip_heuristics(instance)
            instance_before_reset = instance.copy_orig()
            env.seed(self.seed)
            obs, action_set, reward, done, info = env.reset(instance)

            if obs is not None:
                if self.threshold_difficulty is not None:
                    # check difficulty using threshold agent
                    self.threshold_env.seed(self.seed)
                    _obs, _action_set, _reward, _done, _info = self.threshold_env.reset(instance_before_reset.copy_orig())
                    while not _done:
                        _action, _action_idx = self.threshold_agent.action_select(_action_set, _obs)
                        # _action = _action_set[_action]
                        _obs, _action_set, _reward, _done, _info = self.threshold_env.step(_action)
                        if _info['num_nodes'] > self.threshold_difficulty:
                            # already exceeded threshold difficulty
                            break
                    # m = self.threshold_env.model.as_pyscipopt()
                    # if m.getNTotalNodes() <= self.threshold_difficulty:
                    if _info['num_nodes'] <= self.threshold_difficulty:
                        # can give instance to agent to learn on
                        self.curr_instance = instance_before_reset.copy_orig()
                        return obs, action_set, reward, done, info, instance_before_reset
                else:
                    self.curr_instance = instance_before_reset.copy_orig()
                    return obs, action_set, reward, done, info, instance_before_reset

            counter += 1
            if counter > max_attempts:
                raise Exception('Unable to generate valid instance after {} attempts.'.format(max_attempts))

    def step_env(self, observation):
        action, action_idx = self.agent.action_select(observation)
        observation, action_set, reward, done, info = self.env.step(action)
        return observation, action_set, reward, done, info

    def update_episode_log(self, episode_stats):
        for key, val in episode_stats.items():
            self.episodes_log[key].append(val)
        self.episodes_log['episode_counter'] = self.episode_counter
        self.episodes_log['batch_counter'] = self.batch_counter

    def get_episode_log_str(self):
        '''Returns string logging end of episode.'''
        log_str = 'Episode {}'.format(self.episode_counter)
        log_str += ' | Run time: {} s'.format(round(self.episodes_log['episode_run_time'][-1], 3))
        log_str += ' | Nodes: {}'.format(self.episodes_log['num_nodes'][-1])
        log_str += ' | LP iters: {}'.format(self.episodes_log['lp_iterations'][-1])
        log_str += ' | Solve time: {} s'.format(round(self.episodes_log['solving_time'][-1], 3))
        log_str += ' | Total return: {}'.format(round(self.episodes_log['R'][-1], 3))
        return log_str

    def greedy_rollout(self):
        self.agent.eval() # put into evaluation mode
        self.baseline_agent.eval()
        agents = {'agent': self.agent,
                  'baseline_agent': self.baseline_agent}
        envs = {'agent': self.env,
                'baseline_agent': self.baseline_env}
        metrics = [self.agent_reward]

        validator = ReinforcementLearningValidator(agents=agents,
                                                   envs=envs,
                                                   instances=self.instances,
                                                   metrics=metrics,
                                                   seed=0,
                                                   turn_off_heuristics=self.turn_off_heuristics,
                                                   min_threshold_difficulty=None,
                                                   max_threshold_difficulty=self.threshold_difficulty,
                                                   threshold_env=self.threshold_env,
                                                   threshold_agent=self.threshold_agent,
                                                   episode_log_frequency=float('inf'),
                                                   path_to_save=None,
                                                   checkpoint_frequency=float('inf'))
        validator.test(self.kwargs['greedy_rollout_evaluations'])

        agent_score = np.mean([np.sum(rewards) for rewards in validator.episodes_log['agent'][self.agent_reward]])
        baseline_agent_score = np.mean([np.sum(rewards) for rewards in validator.episodes_log['baseline_agent'][self.agent_reward]])

        self.agent.train() # put agent back into training mode

        if agent_score > baseline_agent_score:
            # agent better, should update baseline agent to agent
            return True, agent_score, baseline_agent_score
        else:
            # agent worse, should not update baseline agent
            return False, agent_score, baseline_agent_score

        # if self.agent_reward in ['num_nodes', 'lp_iterations', 'solving_time', 'primal_integral', 'dual_integral', 'primal_dual_integral']:
            # # lower is better
            # if agent_score < baseline_agent_score:
                # # agent better, should update baseline agent to agent
                # return True, agent_score, baseline_agent_score
            # else:
                # # agent worse, should not update baseline agent
                # return False, agent_score, baseline_agent_score
        # else:
            # # higher is better
            # if agent_score > baseline_agent_score:
                # # agent better, should update baseline agent to agent
                # return True, agent_score, baseline_agent_score
            # else:
                # # agent worse, should not update baseline agent
                # return False, agent_score, baseline_agent_score

    def sampled_rollout(self, actions=None, action_sets=None, greedy=False):
        # # DEBUG
        # base_m = self.env.model.as_pyscipopt()
        # print(f'base_m before rollout num nodes: {base_m.getNTotalNodes()} | best primal bound: {base_m.getPrimalbound()} | best dual bound: {base_m.getDualbound()} | primal-dual gap: {base_m.getGap()}')

        if greedy:
            # put agent into eval mode -> acts deterministically
            self.agent.eval()
        else:
            # ensure agent is in train mode -> acts stochastically
            self.agent.train()

        beam_env = EcoleBranching(observation_function=self.env.str_observation_function,
                                  information_function=self.env.str_information_function,
                                  reward_function=self.env.str_reward_function,
                                  scip_params=self.env.str_scip_params)
        beam_env.seed(self.seed)
        
        if actions is not None:
            # get rollout beam instance to same stage as agent's base env
            beam_obs, beam_action_set, beam_reward, beam_done, beam_info = beam_env.reset(self.curr_instance.copy_orig())
            for action, action_set in zip(actions, action_sets):
                beam_obs, beam_action_set, beam_reward, beam_done, beam_info = beam_env.step(action_set[action.item()])
                # beam_obs, beam_action_set, beam_reward, beam_done, beam_info = beam_env.step(action)
            # # DEBUG
            # beam_m = beam_env.model.as_pyscipopt()
            # print(f'beam_m before rollout num nodes: {beam_m.getNTotalNodes()} | best primal bound: {beam_m.getPrimalbound()} | best dual bound: {beam_m.getDualbound()} | primal-dual gap: {beam_m.getGap()}')
        else:
            # discard search tree and initialise env from copied scip instance
            beam_m = pyscipopt.Model(sourceModel=self.env.model.as_pyscipopt(), origcopy=False, globalcopy=True, createscip=True)
            beam_m = ecole.scip.Model.from_pyscipopt(beam_m)
            beam_obs, beam_action_set, beam_reward, beam_done, beam_info = beam_env.reset(beam_m)
            if beam_done:
                # beam pre-solved with primal heuristic from current state -> agent can solve by doing 1 more branch to add 2 more nodes
                return [-2]

        # sample actions from current state to episode termination to get baseline
        beam_rewards = []
        if self.kwargs['apply_max_steps_to_rollout']:
            max_steps = self.max_steps
        else:
            max_steps = int(1e12)
        for t in range(len(actions), max_steps):
            if self.action_filter_agent is not None:
                beam_action_set = self.filter_actions_with_scores(beam_env.model, beam_done, beam_action_set)
            beam_action, beam_action_idx = self.agent.action_select(beam_action_set, beam_obs)
            # beam_action = beam_action_set[beam_action.item()]
            beam_obs, beam_action_set, beam_reward, beam_done, beam_info = beam_env.step(beam_action)
            beam_rewards.append(beam_reward[self.agent_reward])
            if beam_done:
                break
        if not beam_done:
            # max steps for using RL agent reached
            if self.max_steps_agent is not None:
                # use max_steps agent to solve rest of episode
                while not beam_done:
                    beam_action, beam_action_idx = self.max_steps_agent.action_select(beam_action_set, beam_obs)
                    # beam_action = beam_action_set[beam_action.item()]
                    beam_obs, beam_action_set, beam_reward, beam_done, beam_info = beam_env.step(beam_action)
                    beam_rewards.append(beam_reward[self.agent_reward])

        # # DEBUG
        # beam_m = beam_env.model.as_pyscipopt()
        # print(f'beam_m after rollout num nodes: {beam_m.getNTotalNodes()} | best primal bound: {beam_m.getPrimalbound()} | best dual bound: {beam_m.getDualbound()} | primal-dual gap: {beam_m.getGap()}')

        # ensure agent is returned to train mode
        self.agent.train()

        return beam_rewards




        # beam_rewards = []
        # curr_state = self.env.model.as_pyscipopt()

        # beam_m = pyscipopt.Model(sourceModel=curr_state, origcopy=False, globalcopy=True, createscip=True)
        # beam_env = EcoleBranching(observation_function=self.env.str_observation_function,
                                  # information_function=self.env.str_information_function,
                                  # reward_function=self.env.str_reward_function,
                                  # scip_params=self.env.str_scip_params)
        # beam_env = EcoleBranching()
        # beam_m = ecole.scip.Model.from_pyscipopt(beam_m)
        # beam_m = turn_off_scip_heuristics(beam_m)
        # beam_env.seed(self.seed)
        # beam_obs, beam_action_set, beam_reward, beam_done, beam_info = beam_env.reset(beam_m)
        # beam_m = beam_env.model.as_pyscipopt()
        # if beam_done:
            # # beam pre-solved with primal heuristic from current state -> agent can solve by doing 1 more branch to add 2 more nodes
            # return [-2]
        # else:
            # while not beam_done:
                # beam_action = self.agent.action_select(beam_action_set, beam_obs)
                # beam_action = beam_action_set[beam_action.item()]
                # beam_obs, beam_action_set, beam_reward, beam_done, beam_info = beam_env.step(beam_action)
                # beam_rewards.append(beam_reward[self.agent_reward])
            # return beam_rewards



        # # calc discounted return at each step
        # beam_returns, beam_R = [], 0
        # for beam_r in beam_rewards[::-1]:
            # beam_R = beam_r + (self.gamma * beam_R)
            # beam_returns.insert(0, beam_R)
        # beam_m = beam_env.model.as_pyscipopt()
        # print(f'beam_m after rollout num nodes: {beam_m.getNTotalNodes()} | best primal bound: {beam_m.getPrimalbound()} | best dual bound: {beam_m.getDualbound()} | primal-dual gap: {beam_m.getGap()}')
        # # return discounted return at agent's current step
        # print('discounted beam return at agent curr step: {}'.format(beam_returns[0]))
        # return beam_returns[0]





    # def scale_rewards(self, rewards):
        # '''Scales each reward in rewards to sense*(0, 1) range depending on fraction
        # by which reward contributed to total overall reward (return).
        # '''
        # scaled_rewards = []
        # if len(rewards) == 1:
            # # reward contributed to 100% of overall reward
            # if rewards[0] < 0:
                # return [-1]
            # else:
                # return [1]
        # total_reward = np.sum(np.abs(rewards))
        # for reward in rewards:
            # if reward < 0:
                # sense = -1
            # else:
                # sense = 1
            # reward_frac = sense * (abs(reward) / total_reward)
            # scaled_rewards.append(reward_frac)
        # return scaled_rewards




    # def scale_rewards_sequentially(self, rewards):
        # '''Using difference between initial and final reward, sets
        # reward at each step as how much the previous reward was changed by 
        # as a fraction of the difference in initial and dinal reward such
        # that reward at each step is between 0 and 1 and all rewards sum to 1.
        # '''
        # reward_difference = rewards[0] - rewards[-1]
        # scaled_rewards = [0]
        # for idx in range(1, len(rewards)):
            # scaled_rewards.append((rewards[idx-1]-rewards[idx])/reward_difference)
        # return scaled_rewards



    # def scale_rewards_totally(self, rewards):
        # '''Using the total reward, sets reward at each step as how much it 
        # contributed to the total reward by such that reward at each step is 
        # between 0 and 1 and all rewards sum to 1.
        # '''
        # pass



    def step_optimizer(self, baselines=None):
        returns = {episode: [] for episode in self.experiences.keys()}
        for episode in self.experiences.keys():
            rewards = self.experiences[episode]['rewards']
            # calc discounted returns for each step in episode
            R = 0
            for r in rewards[::-1]:
                R = r + (self.gamma * R)
                returns[episode].insert(0, R)

        if self.baseline is None:
            # no baselines, no need to scale discounted returns
            pass
            
        elif self.baseline == 'mean':        
            # use mean episode return to scale episode step discounted returns

            # scale returns w/ baseline to reduce variance
            scaled_returns_dict = {episode: [] for episode in self.experiences.keys()}
            # use mean of agent returns as baseline
            flattened_returns = torch.tensor([r for rs in list(returns.values()) for r in rs])
            scaled_returns = (flattened_returns - flattened_returns.mean()) / (flattened_returns.std() + self.eps)
            # put scaled returns back into dict
            idx = 0
            for episode in self.experiences.keys():
                episode_length = len(self.experiences[episode]['rewards'])
                scaled_returns_dict[episode] = scaled_returns[idx:idx+episode_length]
                idx += episode_length
            returns = scaled_returns_dict

        else:
            # use baseline return to scale episode step discounted returns
            scaled_returns_dict = {episode: [] for episode in self.experiences.keys()}
            for episode in self.experiences.keys():
                baseline_return = self.baselines[episode]
                if type(baseline_return) == list:
                    # have stored baseline for each step
                    baselines = self.baselines[episode]
                else:
                    # have stored baseline for whole episode, must re-calc for each step
                    num_agent_steps = len(self.experiences[episode]['rewards'])
                    baselines = [baseline_return/num_agent_steps for _ in range(num_agent_steps)]
                for R, baseline in zip(returns[episode], baselines):
                    scaled_returns_dict[episode].append(R-baseline)
            returns = scaled_returns_dict

        # calc policy loss for each (obs, action, reward) experience at each step in each episode
        self.optimizer.zero_grad()
        for episode in self.experiences.keys():
            # use discounted reward to optimise for each step taken by RL agent
            for t in range(min(len(self.experiences[episode]['rewards']), self.max_steps)):
                logits = self.agent.get_logits(self.experiences[episode]['observations'][t])
                probs = self.agent.get_action_probs(self.experiences[episode]['action_sets'][t], logits)
                m = torch.distributions.Categorical(probs) # init discrete categorical distribution from softmax probs
                log_prob = m.log_prob(self.experiences[episode]['actions'][t])
                policy_loss = -log_prob * returns[episode][t]
                # calc gradients of policy loss w.r.t. networks parameters
                policy_loss.backward()
        # update networks parameters
        self.optimizer.step()

    def filter_actions_with_scores(self, model, done, action_set):
        # get action scores
        action_scores = self.action_filter_agent.extract(model, done)
        # filter invalid action scores
        action_scores = [action_scores[idx] for idx in action_set]
        # get score treshold below which actions will not be presented to agent
        score_threshold = np.percentile(action_scores, self.action_filter_percentile)
        # update action set
        action_set = [action_set[idx] for idx in range(len(action_set)) if action_scores[idx] >= score_threshold]
        return action_set



    def run_episodes(self, batch_size):
        self.baselines = {e: [] for e in range(batch_size)}
        self.experiences = {e: {'observations': [], 'actions': [], 'action_sets': [], 'rewards': []} for e in range(batch_size)}
        for episode in range(batch_size):
            self.run_episode(episode)
        # finished batch of episodes
        self.batch_counter += 1

    def run_episode(self, episode):
        '''Runs one episode.

        Args:
            episode (int): ID for episode within a batch. Used for dict keys.
        '''
        episode_stats = defaultdict(lambda: 0)
        # episode_stats['action_probabilities'] = [] # track action probabilities for each step
        obs, action_set, reward, done, info, instance = self.reset_env(env=self.env)
        if self.baseline_env is not None:
            # reset baseline with same instance as agent
            if self.baseline == 'sb' or self.baseline == 'pc':
                self.baseline_agent.before_reset(instance)
            self.baseline_env.seed(self.seed)
            baseline_obs, baseline_action_set, baseline_reward, baseline_done, baseline_info = self.baseline_env.reset(instance.copy_orig())

        ep_start = time.time()
        # with torch.set_grad_enabled(self.optimizer is not None):
        with torch.no_grad():
            for t in range(self.max_steps): # don't infinite loop while training

                if self.baseline == 'sr':
                    # check if should do sampled rollout from current agent step

                    if self.episode_counter % self.kwargs['sampled_rollout_frequency'] == 0 or self.prev_beam_baseline is None:
                        # stochastically sample actions from current state to episode termination to get baseline
                        beam_baselines = []
                        for beam in range(self.kwargs['sampled_rollout_beams']):
                            beam_rewards = self.experiences[episode]['rewards'] + self.sampled_rollout(actions=self.experiences[episode]['actions'], action_sets=self.experiences[episode]['action_sets'])
                            # if self.scale_episode_rewards:
                                # beam_rewards = self.scale_rewards_sequentially(beam_rewards)

                            # calc discounted return at each step
                            beam_returns, beam_R = [], 0
                            for beam_r in beam_rewards[::-1]:
                                beam_R = beam_r + (self.gamma * beam_R)
                                beam_returns.insert(0, beam_R)

                            # use beam's discounted return at agent's current step as baseline
                            beam_baselines.append(beam_returns[t])

                        # use mean of discounted returns across all beams as baseline for current step
                        self.prev_beam_baseline = np.mean(beam_baselines) # update previous beam baseline tracker
                        self.baselines[episode].append(self.prev_beam_baseline)
                    else:
                        # use previously sampled rollout baseline as baseline for this step
                        self.baselines[episode].append(self.prev_beam_baseline)

                if self.action_filter_agent is not None:
                    action_set = self.filter_actions_with_scores(self.env.model, done, action_set)

                # get agent action
                action, action_idx = self.agent.action_select(action_set, obs)
                self.experiences[episode]['observations'].append(self.agent.obs)
                self.experiences[episode]['action_sets'].append(self.agent.action_set)
                self.experiences[episode]['actions'].append(self.agent.action_idx) # store raw nn action for learning later
                # action = action_set[action.item()]

                # take step in environments
                obs, action_set, reward, done, info = self.env.step(action)
                self.experiences[episode]['rewards'].append(reward[self.agent_reward])
                episode_stats['primal_integral'] += abs(reward['primal_integral'])
                episode_stats['dual_integral'] += abs(reward['dual_integral'])
                episode_stats['primal_dual_integral'] = episode_stats['primal_integral'] - episode_stats['dual_integral']
                episode_stats['num_steps'] += 1

                if done:
                    break

            if not done:
                # reached max_steps
                print('Reached max steps agent at t={}. Num nodes: {}'.format(t, info['num_nodes']))
                if self.max_steps_agent is not None:
                    # use max_steps agent to solve rest of episode
                    while not done:
                        self.experiences[episode]['observations'].append(obs)
                        self.experiences[episode]['action_sets'].append(action_set)
                        # get action
                        action, action_idx = self.max_steps_agent.action_select(action_set, obs)
                        self.experiences[episode]['actions'].append(self.agent.action_idx) # store raw nn action for learning later
                        # action = action_set[action.item()]
                        # take step in environments
                        obs, action_set, reward, done, info = self.env.step(action)
                        self.experiences[episode]['rewards'].append(reward[self.agent_reward])

            if self.baseline is not None:
                if self.baseline in ['mean', 'sr']:
                    # mean -> no need to use separate agent
                    # sr -> already calculated baseline
                    pass
                else:
                    # get baseline total returns
                    while not baseline_done:
                        if self.baseline in ['sb', 'pc']:
                            baseline_action, baseline_action_idx = self.baseline_agent.action_select(baseline_action_set, self.baseline_env.model, baseline_done)
                        else:
                            baseline_action, baseline_action_idx = self.baseline_agent.action_select(baseline_action_set, baseline_obs)
                        # baseline_action = baseline_action_set[baseline_action]
                        baseline_obs, baseline_action_set, baseline_reward, baseline_done, baseline_info = self.baseline_env.step(baseline_action)
                    self.baselines[episode] = baseline_reward[self.agent_reward]

            # finished episode
            ep_end = time.time()
            episode_stats['R'] = np.sum(self.experiences[episode]['rewards']) # episode return
            episode_stats['episode_run_time'] = ep_end - ep_start
            episode_stats['num_nodes'] = info['num_nodes']
            episode_stats['solving_time'] = info['solving_time']
            episode_stats['lp_iterations'] = info['lp_iterations']
            if self.baseline is not None and self.baseline != 'mean':
                episode_stats['baselines'] = self.baselines[episode]
            self.update_episode_log(episode_stats)

            if self.baseline == 'gr':
                if self.episode_counter % self.kwargs['greedy_rollout_frequency'] == 0 and self.episode_counter != 0:
                    # perform greedy rollout -> update baseline if agent better
                    update_baseline, agent_score, baseline_agent_score = self.greedy_rollout()
                    print('Greedy rollout | Agent score: {} | Baseline agent score: {} | Update baseline: {}'.format(agent_score, baseline_agent_score, update_baseline))
                    if update_baseline:
                        self.baseline_agent.policy_network.load_state_dict(self.agent.policy_network.state_dict())
                        if self.kwargs['threshold_agent'] == 'baseline_agent' and self.threshold_env is not None:
                            # update threshold agent as well
                            self.threshold_agent.policy_network.load_state_dict(self.agent.policy_network.state_dict())

            if self.episode_counter % self.episode_log_frequency == 0 and self.episode_log_frequency != float('inf'):
                print(self.get_episode_log_str())
            if self.episode_counter % self.checkpoint_frequency == 0:
                self.save_checkpoint({'episodes_log': self.episodes_log})

            self.episode_counter += 1

    def train(self, num_episodes):
        if num_episodes < self.batch_size:
            raise Exception('Cannot have num episodes {} < batch size {}.'.format(num_episodes, self.batch_size))
        self.train_start = time.time()
        self.episode_counter, self.batch_counter = 0, 0
        self.episodes_log['episode_counter'] = 0
        self.episodes_log['batch_counter'] = 0
        num_batches = int(num_episodes/self.batch_size)
        for _ in range(num_batches):
            self.run_episodes(self.batch_size)
            with torch.set_grad_enabled(self.optimizer is not None):
                self.step_optimizer(self.baselines)
        self.train_end = time.time()
        self.save_checkpoint({'episodes_log': self.episodes_log})
        print('Trained agent {} on {} episodes in {} s.'.format(self.agent.name, num_batches*self.batch_size, round(self.train_end-self.train_start, 3)))

    def init_episodes_log(self):
        # lists for logging stats at end of each episode
        episodes_log = defaultdict(list)

        # other learner params
        episodes_log['agent_name'] = self.agent.name
        episodes_log['agent_reward'] = self.agent_reward
        episodes_log['agent_device'] = self.agent.device
        episodes_log['max_steps'] = self.max_steps
        episodes_log['batch_size'] = self.batch_size
        episodes_log['baseline'] = self.baseline
        episodes_log['lr'] = self.lr
        episodes_log['gamma'] = self.gamma
        episodes_log['turn_off_heuristics'] = self.turn_off_heuristics
        episodes_log['threshold_difficulty'] = self.threshold_difficulty
        episodes_log['learner_name'] = self.name
        episodes_log['overfit_instance'] = self.overfit_instance
        # episodes_log['max_log_probs'] = self.max_log_probs

        return episodes_log
