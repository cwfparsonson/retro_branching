bfs_scip_params = {
                   'separating/maxrounds': 0, # separate (cut) only at root node
                   'presolving/maxrestarts': 0, # disable solver restarts
                   'limits/time': 3600, # solver time limit
                   
                   'nodeselection/dfs/stdpriority': -536870912,
                   'nodeselection/dfs/memsavepriority': -536870912,
                   
                   # RestartDFS is different from DFS (https://www.mdpi.com/2673-2688/2/2/10), disable
                   'nodeselection/restartdfs/stdpriority': -536870912,
                   'nodeselection/restartdfs/memsavepriority': -536870912,
                   'nodeselection/restartdfs/selectbestfreq': 0,
                   
                   'nodeselection/bfs/stdpriority': 1073741823,
                   'nodeselection/bfs/memsavepriority': 536870911,
                   
                   'nodeselection/breadthfirst/stdpriority': -536870912,
                   'nodeselection/breadthfirst/memsavepriority': -536870912,
                   
                   'nodeselection/estimate/stdpriority': -536870912,
                   'nodeselection/estimate/memsavepriority': -536870912,
                   
                   'nodeselection/hybridestim/stdpriority': -536870912,
                   'nodeselection/hybridestim/memsavepriority': -536870912,
                   
                   'nodeselection/uct/stdpriority': -536870912,
                   'nodeselection/uct/memsavepriority': -536870912,
                  }