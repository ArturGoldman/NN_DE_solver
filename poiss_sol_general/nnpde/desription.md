# Files

**metrics.py:** Calculates MSE between weights, while training JacobyWithConv

**problems.py:** Defines classes
- DirichletProblem: initialises problem on given grid and solves it using regular iterative_methods.jacoby_method from random approximation
    - .compute_solution: computes solution for same setting, but using H_method (with trained H net)

- after_solver: extracts learned H from JacobyWithConv. Applies H method to given task.
    - .calc_sec - calculate stein CV using approximation formulas.

**solve_area.py:** Defines classes

- Grid_Object: defines discrete grid
    - .get_logs - calculates grad_log of given distribution for every point on the grid. returns array of size [1, n_dims, [N]]. So that grad_log[1, i] is \partial log/\partial x_i in every point on gird
    - .set_borders - creates n-dim tensor, so that it is 0 on the inside, and random on border. B_idx is indicator of inner region. B specifies f used for training initial problem instances
    - .create_f_grid - calculates value of passed function on every node in grid
- Solve_Area: trains net H
    - .train_model - trains model
    - .solve_setting - retraining for specified function. sets right part of equation
    - .get_cv - gets stein CV value for given sample.


**model.py** Defines classes:
- _ConvNet_ - n dimensional convolutional NN
- JacobyWithConv: method of solving diff eq. Trains H over true problem instances to fit them better to correct answer

**iterative_methods.py** Defines functions:

- _jacobi_iteration_step_
- jacobi_method
- H_method

