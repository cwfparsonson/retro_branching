======================================
:space_invader: Retro Branching :herb:
======================================

--------------------------------------------------------------------------------------------------------------------------------------------

Implementation of retro branching as reported in 'Reinforcement Learning for Branch-and-Bound Optimisation using Retrospective Trajectories'

--------------------------------------------------------------------------------------------------------------------------------------------


Setup
=====

Open your command line. Change the current working directory to the location where you want to clone this project, and run::

    $ git clone https://github.com/cwfparsonson/retro_branching

In the project's root directory, run::

    $ python setup.py install

Then, still in the root directory, install the required packages with conda (env name defined at top of .yaml file)::

    $ conda env create -f requirements/default.yaml


Using the Retro Branching Method in your Own Code
=================================================

Although tracking the state of the B&B search tree is not natively supported by ``SCIP``, ``PySCIPOpt``, or ``Ecole``, doing so is relatively straightforward. We have
provided two standalone notebook tutorials; one on tracking the B&B search tree (`notebooks/tutorials/tutorial_1_tracking_the_bnb_search_tree.ipynb <https://github.com/cwfparsonson/retro_branching/blob/master/notebooks/tutorials/tutorial_1_tracking_the_bnb_search_tree.ipynb>`_), 
and another on using the search tree to construct retrospective trajectories for your own agents (`notebooks/tutorials/tutorial_2_retro_branching.ipynb <https://github.com/cwfparsonson/retro_branching/blob/master/notebooks/tutorials/tutorial_2_retro_branching.ipynb>`_).


Re-Running the Paper's Experiments
===============================

For the following examples, consider that you want to save your data to '.' (replace as desired). To begin, navigate to the root of where you have saved this ``retro_branching`` repository folder ready to run the commands from your command line.

N.B. To control the combinatorial optimisation instance in question and its corresponding key word arguments, refer to the following table:

=============================   ========================
CO Class                        Kwargs 
=============================   ========================
set_covering                    n_rows n_cols
maximum_independent_set         n_nodes
combinatorial_auction           n_items n_bids
capacitated_facility_location   n_customers n_facilities 
=============================   ========================


Training
--------

To run the training experiments on device 'cuda:0' (replace as desired), run the following commands:

Reinforcement Learning
~~~~~~~~~~~~~~~~~~~~~

Refer to the following table to see which ``--config-name`` to use for training each RL agent:

========   =============
RL Agent   config-name
========   =============
Original   original.yaml
Retro      retro.yaml
FMSTS      fmsts.yaml
========   =============

E.g. Train Retro on set covering instances with 165 rows and 230 columns::

    $ python experiments/dqn_trainer.py --config-path=configs --config-name=retro.yaml experiment.device=cuda:0 learner.path_to_save=. instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230

For an example of how to interact with and visualise the saved training data, see `notebooks/paper/training_curves.ipynb <https://github.com/cwfparsonson/retro_branching/blob/master/notebooks/paper/training_curves.ipynb>`_
    
Imitation Learning
~~~~~~~~~~~~~~~~~~

Generate labelled strong branching experiences on parallel CPUs for set covering instances with 165 rows and 230 columns::

    $ python experiments/gen_imitation_data.py --config-path=configs --config-name=gen_imitation_data.yaml experiment.path_to_save=. instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230
    
Train IL on the generated strong branching experiences::

    $ python experiments/imitation_trainer.py --config-path=configs --config-name=il.yaml experiment.device=cuda:0 experiment.path_to_save=. experiment.path_to_load_imitation_data=. instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230


Testing
-------

Download the validation instances from Google drive to a ``retro_branching_paper_validation_instances`` folder::

    $ gdown https://drive.google.com/file/d/1knhbVEM0N5PbYU653ilk3gu6FbVRVSG-
    
Download the trained ML agents from Google drive to a ``retro_branching_paper_validation_agents`` folder::

    $ gdown https://drive.google.com/file/d/1knhbVEM0N5PbYU653ilk3gu6FbVRVSG-

Run a trained RL agent on the appropriate validation instances (set covering instances with 500 rows and 1000 columns have the Retro, Original, FMSTS, and IL agents available, all other CO instances have only the Retro and IL agents available) and save the validation results to an ``rl_validator`` folder in the appropriate agent directory of the ``retro_branching_paper_validation_agents`` folder from above::

    $ python experiments/validator.py --config-path=configs --config-name=validator.yaml environment.observation_function=43_var_features environment.scip_params=gasse_2019 instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230 experiment.agent_name=retro experiment.path_to_load_agent=./retro_branching_paper_validation_agents experiment.path_to_load_instances=./retro_branching_paper_validation_instances experiment.path_to_save=./retro_branching_paper_validation_agents/ experiment.device=cuda:0

Run a trained RL agent in a DFS node selection environment to get e.g. the FMSTS-DFS agent from the paper::

    $ python experiments/validator.py --config-path=configs --config-name=validator.yaml environment.observation_function=43_var_features environment.scip_params=dfs instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230 experiment.agent_name=fmsts experiment.path_to_load_agent=./retro_branching_paper_validation_agents experiment.path_to_load_instances=./retro_branching_paper_validation_instances experiment.path_to_save=./retro_branching_paper_validation_agents/ experiment.device=cuda:0
    
Run a trained IL agent on the appropriate validation instances (i.e. same as RL agents but with 19 feature observation)::

    $ python experiments/validator.py --config-path=configs --config-name=validator.yaml environment.observation_function=default environment.scip_params=gasse_2019 instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230 experiment.agent_name=il experiment.path_to_load_agent=./retro_branching_paper_validation_agents experiment.path_to_load_instances=./retro_branching_paper_validation_instances experiment.path_to_save=./retro_branching_paper_validation_agents/ experiment.device=cuda:0
    
Run a strong branching agent::

    $ python experiments/validator.py --config-path=configs --config-name=validator.yaml environment.scip_params=gasse_2019 instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230 experiment.agent_name=strong_branching experiment.path_to_load_instances=./retro_branching_paper_validation_instances experiment.path_to_save=./retro_branching_paper_validation_agents/ experiment.device=cpu
    
Run a pseudocost branching agent::

    $ python experiments/validator.py --config-path=configs --config-name=validator.yaml environment.scip_params=gasse_2019 instances.co_class=set_covering instances.co_class_kwargs.n_rows=165 instances.co_class_kwargs.n_cols=230 experiment.agent_name=pseudocost_branching experiment.path_to_load_instances=./retro_branching_paper_validation_instances experiment.path_to_save=./retro_branching_paper_validation_agents/ experiment.device=cpu

The above validation runs will each save an ``episodes_log.pkl`` file. Below is an example of how to interact with this file in Python:

.. code:: python

    import pickle
    import gzip
    import numpy as np

    file = './retro_branching_paper_validation_agents/set_covering_n_rows_500_n_cols_1000/retro/rl_validator/rl_validator_1/checkpoint_11/episodes_log.pkl'
    with gzip.open(file, 'rb') as f:
        log = pickle.load(f)
    agent_name = log['agent_names'][0]

    # get number of nodes achieved for each instance
    num_nodes_for_each_instance = [np.abs(np.sum(episode_nodes)) for episode_nodes in log[agent_name]['num_nodes']]
    print(f'Per-instance # nodes: {num_nodes_for_each_instance}')
    print(f'All-instances mean # nodes: {np.mean(num_nodes_for_each_instance)}')

For more examples of how to interact with and visualise the saved validation data, see `notebooks/paper/performance_bar_charts.ipynb <https://github.com/cwfparsonson/retro_branching/blob/master/notebooks/paper/performance_bar_charts.ipynb>`_ and `notebooks/paper/winner_plots.ipynb <https://github.com/cwfparsonson/retro_branching/blob/master/notebooks/paper/winner_plots.ipynb>`_


Citing this work
================

TODO: 

- Add citation
