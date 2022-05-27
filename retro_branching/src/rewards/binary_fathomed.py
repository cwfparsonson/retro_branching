from retro_branching.utils import SearchTree

import copy

class BinaryFathomed:
    def __init__(self, not_fathomed=-1, fathomed=0):
        self.not_fathomed = not_fathomed
        self.fathomed = fathomed

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
            return 0

        # update search tree with current model state
        self.tree.update_tree(model)
        
        # collect node stats from children introduced from previous branching decision
        prev_node_child_ids = [child for child in self.tree.tree.successors(self.prev_node_id)]

        # calc reward for previous branching decision
        if len(prev_node_child_ids) > 0:
            # previous branching decision did not fathom sub-tree
            closed_by_agent = False
            score = self.not_fathomed
        else:
            # previous branching decision fathomed sub-tree
            closed_by_agent = True
            score = self.fathomed

        # update tree with effect(s) of branching decision
        self.tree.tree.nodes[self.prev_node_id]['score'] = score
        self.tree.tree.nodes[self.prev_node_id]['closed_by_agent'] = closed_by_agent

        if m.getCurrentNode() is not None:
            # update stats for next step
            self.prev_node = m.getCurrentNode()
            self.prev_node_id = copy.deepcopy(self.prev_node.getNumber())
        else:
            # instance completed, no current focus node
            pass

        return score