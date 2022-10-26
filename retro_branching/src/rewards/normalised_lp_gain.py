from retro_branching.utils import SearchTree
import copy
import math

class NormalisedLPGain:
    def __init__(self, 
                 use_binary_sparse_rewards=False,
                 normaliser='init_primal_bound', 
                 transform_with_log=False,
                 epsilon=None):
        '''
        Args:
            use_binary_sparse_rewards (bool): If True, rather than LP-gain, will simply
                return a 1 on the step that solves the instance and a 0 on all
                other steps. Implemented this here to save writing another class
                which tracks sub-trees.
            normaliser ('init_primal_bound', 'curr_primal_bound'): What to normalise
                with respect to in the numerator and denominator to calculate
                the per-step normalsed LP gain reward.
            transform_with_log: If True, will transform the reward by doing
                reward = sign(reward) * log(1 + |reward|) -> helps reduce variance
                of reward as in https://arxiv.org/pdf/1704.03732.pdf
            epsilon (None, float): If not None, will set score of nodes which
                were pruned, infeasible, or outside of bounds to epsilon (rather than
                0 if added or rather than not considering at all if never added to tree).
                N.B. epsilon should be a small number e.g. 1e-6. N.B.2. Current
                implementation assumes each branching decision results in 2 child
                nodes (regardless of whether or not SCIP stores these in memory).
        '''
        self.use_binary_sparse_rewards = use_binary_sparse_rewards
        self.normaliser = normaliser
        self.transform_with_log = transform_with_log
        self.epsilon = epsilon

    def before_reset(self, model):
        self.prev_node = None
        self.prev_node_id = None
        self.prev_primal_bound = None
        self.init_primal_bound = None

    def extract(self, model, done):
        m = model.as_pyscipopt()

        if self.prev_node_id is None:
            # not yet started, update prev node for next step
            self.prev_node = m.getCurrentNode()
            self.tree = SearchTree(model)
            if self.prev_node is not None:
                self.prev_node_id = copy.deepcopy(self.prev_node.getNumber())
                self.prev_primal_bound = m.getPrimalbound()
                self.init_primal_bound = m.getPrimalbound()
            # return 0  # dsorokin: I think it is bug

        # update search tree with current model state
        self.tree.update_tree(model)
        
        # collect node stats from children introduced from previous branching decision
        prev_node_lb = self.tree.tree.nodes[self.prev_node_id]['lower_bound']
        prev_node_child_ids = [child for child in self.tree.tree.successors(self.prev_node_id)]
        prev_node_child_lbs = [self.tree.tree.nodes[child]['lower_bound'] for child in prev_node_child_ids]

        # calc reward for previous branching decision
        if len(prev_node_child_lbs) > 0:
            # use child lp gains to retrospectively calculate a score for the previous branching decision
            closed_by_agent = False
            if self.use_binary_sparse_rewards:
                # score = 0
                score = -1
            else:
                score = -1
                for child_node_lb in prev_node_child_lbs:
                    if self.normaliser == 'curr_primal_bound':
                        # use primal bound of step branching action was taken
                        score *= (self.prev_primal_bound - child_node_lb) / (self.prev_primal_bound - prev_node_lb)
                    elif self.normaliser == 'init_primal_bound':
                        # use init primal bound
                        score *= (self.init_primal_bound - child_node_lb) / (self.init_primal_bound - prev_node_lb)
                    else:
                        raise Exception(f'Unrecognised normaliser {self.normaliser}')
                if self.epsilon is not None:
                    # consider child nodes which were never added to search tree by SCIP
                    for _ in range(int(2-len(prev_node_child_lbs))):
                        score *= self.epsilon
        else:
            # previous branching decision led to all child nodes being pruned, infeasible, or outside bounds -> don't punish brancher
            closed_by_agent = True
            if self.use_binary_sparse_rewards:
                # score = 1
                score = 0
            else:
                if self.epsilon is not None:
                    score = -1 * (self.epsilon**2)
                else:
                    score = 0

        # update tree with effect(s) of branching decision
        self.tree.tree.nodes[self.prev_node_id]['score'] = score
        self.tree.tree.nodes[self.prev_node_id]['closed_by_agent'] = closed_by_agent

        if m.getCurrentNode() is not None:
            # update stats for next step
            self.prev_node = m.getCurrentNode()
            self.prev_node_id = copy.deepcopy(self.prev_node.getNumber())
            self.prev_primal_bound = m.getPrimalbound()
        else:
            # instance completed, no current focus node
            pass

        if self.transform_with_log:
            sign = math.copysign(1, score)
            score = sign * math.log(1 + abs(score), 10)

        if score < -1:
            print('Score < -1 found.')
            for node in self.tree.tree.nodes():
                print(f'Node {node} lb: {self.tree.tree.nodes[node]["lower_bound"]}')
            raise Exception()
        
        return score