import pulp
import copy
import math
import gym
import time
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import ecole


class LPSolver:
    def __init__(self, solver=None):
        self.solver = solver 
    
    def solve(self, instance):
        if type(instance) == dict:
            _instance = copy.deepcopy(instance['instance'])
        else:
            _instance = copy.deepcopy(instance)
        # print(_instance)
        _ = _instance.solve(solver=self.solver)

        return _instance

class Instance(pulp.LpProblem):
    def __init__(self, name='instance', sense=1):
        '''
        Args:
            name (str)
            sense (int): 1 for minimise, -1 for maximise.
        '''
        super(Instance, self).__init__(name, sense)

    def register_integrality_variables(self):
        '''Store variables with integrality constraint in integrality_variables hash table.'''
        self.integrality_variables = {var.name: var for var in self.variables() if var.cat == 'Integer' or var.cat == 'Binary'}

    def solve_primal(self, instance, solver='round'):
        '''Solves primal by taking dual solution and rounding decision variable assignments.

        Args:
            instance: Instance to solve. If using 'round' solver, must be dual
                instance.
            solver (str): Must be one of: 'round'.
        '''
        if solver == 'round':
            primal_instance = copy.deepcopy(instance)
            for var in primal_instance.variables():
                if var.name in self.integrality_variables:
                    # restore constraint and round
                    var.cat = self.integrality_variables[var.name].cat
                    if self.sense == 1:
                        # minimisation, round vars up to get primal solution
                        var.varValue = min(math.ceil(var.value()), var.upBound)
                    else:
                        # maximisation, round vars down to get primal solution 
                        var.varValue = max(math.floor(var.value()), var.lowBound)

        else:
            raise Exception('Unrecognised primal solver \'{}\''.format(solver))

        return primal_instance

    def solve_dual(self, instance, solver=None):
        '''Solves dual by relaxing all integrality constraints.'''
        solver = LPSolver(solver=solver)
        dual_instance = self.relax_integrality_constraints(instance)
        return solver.solve(dual_instance)

    def relax_integrality_constraints(self, instance):
        '''Relaxes integrality constraints for all decision variables.'''
        _instance = copy.deepcopy(instance)
        for var in _instance.variables():
            if var.cat == 'Integer' or var.cat == 'Binary':
                var.cat = 'Continuous'
        return _instance



class DecisionVariable(pulp.LpVariable):
    def __init__(self, name, lowBound=None, upBound=None, cat='Continuous', e=None):
        super(DecisionVariable, self).__init__(name, lowBound=lowBound, upBound=upBound, cat=cat, e=e)

def lp_sum(vector):
    return pulp.lpSum(vector)
















############################### CUSTOM ENVIRONMENTS #########################


class BranchAndBound(gym.Env):
    def __init__(self, 
                 node_selector, 
                 variable_selector, 
                 primal_assigner=None, 
                 tree_pruner=None, 
                 name='branch_and_bound'):
        self.node_selector = node_selector
        self.variable_selector = variable_selector
        self.primal_assigner = primal_assigner
        self.tree_pruner = tree_pruner
        
    def reset(self, instance, compute_baseline=True, name='branch_and_bound'):
        self.name = name
        self.compute_baseline = compute_baseline

        if self.compute_baseline:
            self.baseline_instance = copy.deepcopy(instance)
            start = time.time()
            self.baseline_instance.solve()
            end = time.time()
            self.baseline_solve_time = end-start

        # register integrality variables
        instance.register_integrality_variables() 
        
        # init env state
        self.state = {'instance': instance,
                      'search_tree': nx.DiGraph()}
        
        # solve dual and primal -> get dual-primal bounds for root node
        dual_instance = instance.solve_dual(instance)
        primal_instance = instance.solve_primal(dual_instance)
        
        # update search tree
        self.state['search_tree'].graph['incumbent_node'] = None # init incumbent node 
        self.state['search_tree'].graph['incumbent_node_history'] = []
        self.state['search_tree'].graph['best_primal_node'] = 0 # init best primal (feasible) solution
        self.state['search_tree'].graph['open_nodes'] = {0: None} # init hash table of open nodes
        self.state['search_tree'].graph['fathomed_nodes'] = {} # init hash table of fathomed nodes
        self.state['search_tree'].graph['infeasible_nodes'] = {} # init hash table of infeasible nodes
        self.state['search_tree'].graph['pruned_nodes'] = {} # init hash table of pruned nodes
        self.add_node_to_search_tree(dual_instance, primal_instance, parent_node=None)
        
        self.reward = self.get_node_primal_dual_gap(0)
        self.done = False
        self.info = {}

        self.init_tracking_metrics()

        return self.state
    
    def check_if_done(self):
        '''finished when all leaf nodes in search tree have been fathomed.'''
        if len(self.state['search_tree'].graph['open_nodes']) == 0:
            return True
        else:
            return False
    
    def check_if_should_fathom(self, dual_instance, primal_instance):
        '''Checks if node should be fathomed and returns reason(s) for fathoming.

        Fathom node if:
            (1) infeasible
            (2) primal bound == dual bound
            (3) variables of dual solution w/ integrality constraint are all integers
            (4) dual bound is worse than incumbent's primal bound
        '''
            
        fathom, reasons = False, []
            
        if len(self.state['search_tree'].nodes) == 0:
            # no nodes added yet
            return fathom, reasons
            
        if dual_instance.status == -1:
            # infeasible
            fathom = True
            reasons.append('infeasible')
            
        if dual_instance.objective.value() == primal_instance.objective.value():
            fathom = True
            reasons.append('p==d')
            
        if self.is_integrality_satisfied(dual_instance):
            fathom = True
            reasons.append('integrality')
            
        incumbent = self.state['search_tree'].graph['incumbent_node']
        if incumbent is not None:
            if self.check_if_solution_bounds_outside_incumbent(dual_instance):
                fathom = True
                reasons.append('bounds')
            
        return fathom, reasons
    
    def fathom_node(self, node, reasons):
        self.state['search_tree'].nodes[node]['fathomed'] = True
        self.state['search_tree'].graph['fathomed_nodes'][node] = None
        if 'infeasible' in reasons:
            # fathoming node since is infeasible
            self.state['search_tree'].graph['infeasible_nodes'][node] = None
            # self.state['search_tree'].graph['pruned_nodes'][node] = None
        # if 'bounds' in reasons:
            # # fathoming node since outside incumbent primal bounds
            # self.state['search_tree'].graph['pruned_nodes'][node] = None
        del self.state['search_tree'].graph['open_nodes'][node]


    def prune_node(self, node):
        self.state['search_tree'].nodes[node]['pruned'] = True
        self.state['search_tree'].graph['pruned_nodes'][node] = None
        del self.state['search_tree'].graph['open_nodes'][node]
    
    def check_if_solution_bounds_outside_incumbent(self, instance):
        incumbent = self.state['search_tree'].graph['incumbent_node']
        if self.state['instance'].sense == 1:
            # minimisation -> lower better
            if instance.objective.value() > self.state['search_tree'].nodes[incumbent]['primal_bound']:
                # outside incumbent primal bounds
                return True
        else:
            # maximisation -> higher better
            if instance.objective.value() < self.state['search_tree'].nodes[incumbent]['primal_bound']:
                # outside incumbent primal bounds
                return True
        return False

    def check_if_solution_better_than_incumbent(self, instance):
        if not self.is_integrality_satisfied(instance):
            # infeasible, cannot be better than incumbent
            return False

        incumbent = self.state['search_tree'].graph['incumbent_node']
        if self.state['instance'].sense == 1:
            # minimisation -> lower better
            if instance.objective.value() < self.state['search_tree'].nodes[incumbent]['dual_bound']:
                # better than incumbent
                return True
        else:
            # maximisation -> higher better
            if instance.objective.value() > self.state['search_tree'].nodes[incumbent]['dual_bound']:
                # better than incumbent
                return True
        return False

        
    def is_integrality_satisfied(self, instance):
        '''Checks if all decision variable assignments with integrality constraint are integers.'''
        if instance.status == -1:
            # infeasible
            return False
        
        satisfied = True
        for var in instance.variables():
            if var.name in instance.integrality_variables:
                # var has integrality constraint
                if type(var.varValue) == float:
                    if not var.varValue.is_integer():
                        # not an integer
                        satisfied = False
                        break
                    
        return satisfied
            
    def add_node_to_search_tree(self, dual_instance, primal_instance, parent_node, added_constraint=None):  
        node_id = len(self.state['search_tree'].nodes)

        # check if new primal bound found for this subtree
        if parent_node is not None:
            # not root node
            parent_primal_bound = self.state['search_tree'].nodes[parent_node]['primal_bound']
            parent_primal_instance = self.state['search_tree'].nodes[parent_node]['primal_instance']
            if self.is_integrality_satisfied(dual_instance):
                # new primal bound
                primal_bound = primal_instance.objective.value()
            else:
                # same as parent
                primal_bound = copy.deepcopy(parent_primal_bound)
                primal_instance = copy.deepcopy(parent_primal_instance)
        else:
            # is root node
            primal_bound = primal_instance.objective.value()
        
        if parent_node is not None:
            # not root node
            best_primal_node_obj = self.state['search_tree'].nodes[self.state['search_tree'].graph['best_primal_node']]['primal_instance'].objective.value()
            if self.state['instance'].sense == 1:
                # minimisation -> lower better
                if primal_bound < best_primal_node_obj:
                    # new best primal solution found
                    self.state['search_tree'].graph['best_primal_node'] = node_id
            else:
                # maximisation -> higher better
                if primal_bound > best_primal_node_obj:
                    # new best primal solution found
                    self.state['search_tree'].graph['best_primal_node'] = node_id
        
        # add node
        self.state['search_tree'].graph['open_nodes'][node_id] = None
        self.state['search_tree'].add_node(node_id, 
                                           feasible=dual_instance.status,
                                           pruned=False,
                                           dual_bound=dual_instance.objective.value(), 
                                           primal_bound=primal_bound, 
                                           dual_instance=dual_instance, 
                                           primal_instance=primal_instance)
        fathom, reasons = self.check_if_should_fathom(dual_instance, primal_instance)
        if fathom:
            self.fathom_node(node_id, reasons)
        else:
            self.state['search_tree'].nodes[node_id]['fathomed'] = False
        
        # add edge
        if parent_node is None:
            # root node, no parent
            pass
        else:
            if added_constraint is None:
                raise Exception('If adding new edge, must provide added_constraint which led to edge.')
            self.state['search_tree'].add_edge(parent_node, node_id, added_constraint=added_constraint)
            if parent_node in self.state['search_tree'].graph['open_nodes']:
                # parent node is no longer a leaf node -> remove from open nodes
                del self.state['search_tree'].graph['open_nodes'][parent_node]
        
        # if solution satisfies original integrality constraints, check if should update incumbent
        incumbent = self.state['search_tree'].graph['incumbent_node']
        new_incumbent = False
        if self.is_integrality_satisfied(dual_instance):
            # feasible solution, check if better than incumbent
            if incumbent is not None:
                if self.check_if_solution_better_than_incumbent(dual_instance):
                    # set as new incumbent
                    self.state['search_tree'].graph['incumbent_node'] = node_id
                    new_incumbent = True
                else:
                    pass
            else:
                # dont yet have incumbent node, set as incumbent
                self.num_steps_to_first_incumbent = copy.deepcopy(self.step_counter)
                self.state['search_tree'].graph['incumbent_node'] = node_id
                new_incumbent = True
        if new_incumbent:
            # update incumbent history
            self.state['search_tree'].graph['incumbent_node_history'].append(node_id)
            # fathom open nodes with worse bounds than incumbent
            for node in copy.deepcopy(self.state['search_tree'].graph['open_nodes']).keys():
                if node != node_id:
                    inst = self.state['search_tree'].nodes[node]['dual_instance']
                    if self.check_if_solution_better_than_incumbent(inst):
                        # fathom node since no better than incumbent
                        self.fathom_node(node, reasons=['bounds'])
    
    def step(self, node_choice, variable_choice, prune_choice=None):
        # prune tree 
        
        # # select node
        # node_choice = self.node_selector.compute_action(self.state)
        # print('Chosen node: {}'.format(node_choice))
        
        # # select variable
        # variable_choice = self.variable_selector.compute_action(self.state, node_choice)
        # print('Chosen variable: {}'.format(variable_choice))
        
        # branch on variable
        self.state = self.branch(node_choice, variable_choice)

        self.reward = self.get_node_primal_dual_gap(node_choice)

        self.done = self.check_if_done()

        # update tracking metrics
        self.update_tracking_metrics(node_choice)
        
        return [self.state, self.reward, self.done, self.info]

    def init_tracking_metrics(self):
        self.step_counter = 0
        self.time_start = time.time()
        self.primal_dual_gap_evolution = []
        self.dual_bound_evolution = []
        self.primal_bound_evolution = []

    def get_node_primal_dual_gap(self, node):
        if not self.state['search_tree'].nodes[node]['feasible']:
            return float('inf')
        else:
            return abs(self.state['search_tree'].nodes[node]['dual_bound']-self.state['search_tree'].nodes[node]['primal_bound'])

    def update_tracking_metrics(self, node_choice):
        if self.done:
            self.time_end = time.time()

        self.step_counter += 1

        # get lp gap, dual bound, and primal bound of chosen node
        self.primal_dual_gap_evolution.append(self.get_node_primal_dual_gap(node_choice))
        self.dual_bound_evolution.append(self.state['search_tree'].nodes[node_choice]['dual_bound'])
        self.primal_bound_evolution.append(self.state['search_tree'].nodes[node_choice]['primal_bound'])

        # # check if one of chosen node's chidlren led to new incumbent
        # children = self.state['search_tree'].successors(node_choice)
        # incumbent = self.state['search_tree'].graph['incumbent_node']
        # if incumbent in children:
            # # new incumbent found
            # self.primal_dual_gap_evolution.append(0)
            # self.dual_bound_evolution.append(self.state['search_tree'].nodes[incumbent]['dual_bound'])
            # self.primal_bound_evolution.append(self.state['search_tree'].nodes[incumbent]['primal_bound'])

    
    def branch(self, node_choice, variable_choice):
        instance = self.state['search_tree'].nodes[node_choice]['dual_instance'] # primal has integrality constraints
        variables = {var.name: var for var in instance.variables()}
        
        # branch either side of the chosen variable’s relaxed solution value
        f, i = math.modf(variables[variable_choice].varValue)
        branch_one_instance, branch_two_instance = copy.deepcopy(instance), copy.deepcopy(instance)
        for b_one_var, b_two_var in zip(branch_one_instance.variables(), branch_two_instance.variables()):
            if b_one_var.name == variable_choice:
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
        
        # solve dual and primal -> get dual-primal bounds for branches
        branch_one_dual_instance = branch_one_instance.solve_dual(branch_one_instance)
        branch_one_primal_instance = branch_one_instance.solve_primal(branch_one_dual_instance)
        
        branch_two_dual_instance = branch_two_instance.solve_dual(branch_two_instance)
        branch_two_primal_instance = branch_two_instance.solve_primal(branch_two_dual_instance)
        
        # update search tree
        self.add_node_to_search_tree(branch_one_dual_instance, branch_one_primal_instance, parent_node=node_choice, added_constraint='{}\u2264{}'.format(variable_choice, b_one_var.upBound))
        self.add_node_to_search_tree(branch_two_dual_instance, branch_two_primal_instance, parent_node=node_choice, added_constraint='{}\u2265{}'.format(variable_choice, b_two_var.lowBound))
                
        return self.state
    
    def render(self, 
               with_node_labels=True, 
               with_edge_labels=True, 
               node_size=100, 
               edge_width=1,
               fig_scale=1,
               show_fig=True):
        '''Renders current search tree state.'''
        fig = plt.figure(figsize=[10*fig_scale,10*fig_scale])
        pos = graphviz_layout(self.state['search_tree'], prog='dot')
        
        if with_node_labels:
            node_labels = {}
            for node in self.state['search_tree'].nodes:
                if self.state['search_tree'].nodes[node]['feasible'] == -1:
                    node_labels[node] = '{} infeasible'.format(node)
                else:
                    dual_bound = self.state['search_tree'].nodes[node]['dual_bound']
                    primal_bound = self.state['search_tree'].nodes[node]['primal_bound']
                    node_labels[node] = '{} D: {}, P: {}'.format(node, round(dual_bound, 1), round(primal_bound, 1))
        else:
            node_labels = None
        if with_edge_labels:
            edge_labels = {}
            for edge in self.state['search_tree'].edges:
                edge_labels[edge] = self.state['search_tree'].edges[edge]['added_constraint']
        else:
            edge_labels = None
            
        # draw nodes
        node_type_dict = {'instance': list(self.state['search_tree'].nodes),
                          'fathomed': list(self.state['search_tree'].graph['fathomed_nodes'].keys()), 
                          'pruned': list(self.state['search_tree'].graph['pruned_nodes'].keys()), 
                          'incumbent': [self.state['search_tree'].graph['incumbent_node']]}
        node_type_to_colour = {'instance': '#3DA9DB',
                               'fathomed': '#DB3D3D',
                               'incumbent': '#5DDB3D'}
        for node_type in node_type_dict.keys():
            if len(node_type_dict[node_type]) > 0 and node_type_dict[node_type][0] is not None:
                nx.draw_networkx_nodes(self.state['search_tree'], 
                                       pos, 
                                       nodelist=node_type_dict[node_type],
                                       node_size=node_size, 
                                       node_color=node_type_to_colour[node_type],
                                       label=node_type)
        
        # draw edges
        nx.draw_networkx_edges(self.state['search_tree'], 
                               pos,
                               edgelist=list(self.state['search_tree'].edges),
                               edge_color='k',
                               width=edge_width,
                               label='constraint')
        
        plt.legend()
        
        # node and edge labels
        if with_node_labels:
            nx.draw_networkx_labels(self.state['search_tree'], pos, labels=node_labels)
        if with_edge_labels:
            nx.draw_networkx_edge_labels(self.state['search_tree'], pos, edge_labels=edge_labels)
        
        if show_fig:
            plt.show()

        
        return fig

    def summary(self, print_summary=True):
        steps = [i for i in range(self.step_counter)]
        self.num_nodes = len(self.state['search_tree'].nodes)

        # lp gap evolution
        self.primal_dual_gap_evolution_fig = plt.figure()
        plt.plot(steps, self.primal_dual_gap_evolution)
        plt.xlabel('B&B Steps')
        plt.ylabel('Primal-Dual Gap')
        if print_summary:
            plt.show()

        # dual bound evolution
        self.dual_bound_evolution_fig = plt.figure()
        plt.plot(steps, self.dual_bound_evolution)
        plt.xlabel('B&B Steps')
        plt.ylabel('Dual Bound')
        if print_summary:
            plt.show()

        # primal bound evolution
        self.primal_bound_evolution_fig = plt.figure()
        plt.plot(steps, self.primal_bound_evolution)
        plt.xlabel('B&B Steps')
        plt.ylabel('Primal Bound')
        if print_summary:
            plt.show()

        # summary table
        incumbent = self.state['search_tree'].graph['incumbent_node']
        self.summary_dict = {'# Tree Nodes': [self.num_nodes],
                             '# Fathomed Nodes': [len(list(self.state['search_tree'].graph['fathomed_nodes'].keys()))],
                             '# Infeasible Fathomed Nodes': [len(list(self.state['search_tree'].graph['infeasible_nodes'].keys()))],
                             '# Pruned Nodes': [len(list(self.state['search_tree'].graph['pruned_nodes'].keys()))],
                             '# Steps to 1st Incumbent': [self.num_steps_to_first_incumbent],
                             'Obj': [self.state['search_tree'].nodes[incumbent]['dual_bound']],
                             'Solve Time (s)': [round(self.time_end-self.time_start, 4)]}

        if self.compute_baseline:
            self.summary_dict['Baseline Obj'] = self.baseline_instance.objective.value()
            self.summary_dict['Baseline Solve Time (s)'] = self.baseline_solve_time

        self.summary_df = pd.DataFrame(self.summary_dict)
        if print_summary:
            display(self.summary_df)

        
    
class Pruning(gym.Env):
    def __init__(self, name='pruning'):
        self.reset(name=name)
    
    def reset(self, name='pruning'):
        self.name = name

        self.state = None
        self.reward = 0
        self.done = False
        self.info = {}
        
    def step(self, state):
        pass
    
class NodeSelection(gym.Env):
    def __init__(self, name='node_selection'):
        self.reset(name=name)
    
    def reset(self, name='node_selection'):
        self.reset(name=name)

        self.state = None
        self.reward = 0
        self.done = False
        self.info = {}
        
    def step(self, state):
        pass
    
class VariableSelection(gym.Env):
    def __init__(self, name='variable_selection'):
        self.reset(name=name)
    
    def reset(self, name='variable_selection'):
        self.reset(name=name)

        self.state = None
        self.reward = 0
        self.done = False
        self.info = {}
        
    def step(self, state):
        pass






























