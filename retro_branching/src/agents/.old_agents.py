
from collections import defaultdict
import abc
import math
import copy
import numpy as np
import ml_collections
import json












############################ BASE CLASSES ################################
class NodeSelector(abc.ABC):
    def __init__(self, name='node_selector'):
        self.name = name
    
    @abc.abstractmethod
    def compute_action(self, state):
        '''Return the chosen node to expand subtree on.'''     
        raise NotImplementedError()

class VariableSelector(abc.ABC):
    def __init__(self, name='variable_selector'):
        self.name = name
    
    @abc.abstractmethod
    def compute_action(self, state):
        '''Return the chosen variable to branch on.'''     
        raise NotImplementedError()

class TreePruner(abc.ABC):
    def __init__(self, name='tree_pruner'):
        self.name = name
    
    @abc.abstractmethod
    def compute_action(self, state):
        '''Return the chosen nodes to prune from the search tree.'''     
        raise NotImplementedError()

class PrimalAssigner(abc.ABC):
    def __init__(self, name='primal_assigner'):
        self.name = name
    
    @abc.abstractmethod
    def compute_action(self, state):
        '''Return the primal values to assign to the variables.'''     
        raise NotImplementedError()













########################## PRE-BUILT CLASSES ###############################



############################## SEARCHERS ###################################

class BestFirstSearch(NodeSelector):
    def __init__(self, name='BFS'):
        super(BestFirstSearch, self).__init__(name)
    
    def compute_action(self, state):
        '''Use best-first search to choose node to expand on in search tree.
        
        Will choose whichever node is best (i.e. has best dual bound).
        '''     
        incumbent = state['search_tree'].graph['incumbent_node']
        node_to_dual = {}
        # for node in state['search_tree'].nodes:
            # if not state['search_tree'].nodes[node]['fathomed'] and state['search_tree'].out_degree(node)==0:
                # # can consider this node for selection
                # node_to_dual[node] = state['search_tree'].nodes[node]['dual_bound']
        for node in state['search_tree'].graph['open_nodes']:
            # can consider this node for selection
            node_to_dual[node] = state['search_tree'].nodes[node]['dual_bound']
                    
        if state['instance'].sense == 1:
            # minimisation -> lower is better
            best = min(node_to_dual, key=node_to_dual.get)
        else:
            # maximisation -> higher is better
            best = max(node_to_dual, key=node_to_dual.get)
            
        return np.random.choice([var for var in node_to_dual.keys() if var == best])






############################ BRANCHERS ####################################
    
class MostInfeasibleBranching(VariableSelector):
    def __init__(self, name='MIB'):
        super(MostInfeasibleBranching, self).__init__(name)
    
    def compute_action(self, state, node_choice):
        '''Use most infeasible branching to select variable to branch on.
        
        Will choose whichever variable was assigned the most fractional (closest to 0.5)
        value in the dual solution (i.e. when integrality constraint was relaxed).
        '''
        instance = state['search_tree'].nodes[node_choice]['dual_instance'] # get dual instance
        var_to_fractionality = {}
        for var in instance.variables():
            if var.name in instance.integrality_variables:
                # var has integrality constraint -> can consider for branching
                f, i = math.modf(var.varValue)
                if f != 0:
                    # var is not yet an integer -> can consider for branching
                    var_to_fractionality[var.name] = abs(0.5-f)

        return min(var_to_fractionality, key=var_to_fractionality.get)


class FullStrongBranching(VariableSelector):
    def __init__(self, name='FSB'):
        super(FullStrongBranching, self).__init__(name)

    def compute_action(self, state, node_choice):
        '''Use full strong branching to select a variable to branch on.

        Will choose whichever variable would result in biggest good change in the dual bound
        if chosen to branch on. I.e. must solve 2 LPs for each variable (one
        for each child node/sub-MILP problem introduced by branching on
        chosen variable)
        '''
        dual_instance = copy.deepcopy(state['search_tree'].nodes[node_choice]['dual_instance']) # get dual instance
        variables = {_var.name: _var for _var in dual_instance.variables()}
        var_to_dual_delta = {}
        for var in dual_instance.variables():
            if var.name in dual_instance.integrality_variables:
                # var has integrality constraint -> can consider for branching

                # branch either side of the variable under consideration's relaxed solution value
                f, i = math.modf(variables[var.name].varValue)
                branch_one_instance, branch_two_instance = copy.deepcopy(dual_instance), copy.deepcopy(dual_instance)
                for b_one_var, b_two_var in zip(branch_one_instance.variables(), branch_two_instance.variables()):
                    if b_one_var.name == var.name:
                        if b_one_var.lowBound == 0 and b_one_var.upBound == 1:
                            # is a binary variable, branch w/ x_j = 0 and x_j = 1
                            b_one_var.upBound = 0
                            b_two_var.lowBound = 1
                        else:
                            # not binary, can branch either side of dual assigned value
                            if b_one_var.upBound is not None:
                                b_one_var.upBound = min(i, b_one_var.upBound)
                            else:
                                # is inf, current value will be max, change
                                b_one_var.upBound = i
                            if b_two_var.lowBound is not None:
                                b_two_var.lowBound = max(i+1, b_two_var.lowBound)
                            else:
                                # is -inf, current value will be min, change
                                b_two_var.lowBound = i+1
                        break
                # solve dual for each child
                branch_one_dual_instance = branch_one_instance.solve_dual(branch_one_instance)
                branch_two_dual_instance = branch_two_instance.solve_dual(branch_two_instance)

                # store dual bound delta
                b1_dual_delta, b2_dual_delta = abs(branch_one_dual_instance.objective.value()-dual_instance.objective.value()), abs(branch_two_dual_instance.objective.value()-dual_instance.objective.value())
                if state['instance'].sense == 1:
                    # minimisation -> want bigger change
                    var_to_dual_delta[var.name] = max(b1_dual_delta, b2_dual_delta)
                else:
                    # maximisation -> want smaller change
                    var_to_dual_delta[var.name] = min(b1_dual_delta, b2_dual_delta)

        if state['instance'].sense == 1:
            # minimisation -> want bigger change
            best = max(var_to_dual_delta, key=var_to_dual_delta.get)
        else:
            # maximisation -> want smaller change
            best = min(var_to_dual_delta, key=var_to_dual_delta.get)

        return np.random.choice([var for var in var_to_dual_delta.keys() if var == best])


        
        



