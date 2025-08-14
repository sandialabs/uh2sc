"""Generic mathematical solver classes.

NOTE: This file was taken from the WNTR project open source software at: 2:23PM CST
      3/14/2025 by Daniel Villa (dlvilla@sandia.gov) in compliance with the open-source
      WNTR license. Since then, it has been editted as is allowed by the license.

      https://github.com/USEPA/WNTR/edit/main/wntr/sim/solvers.py

      Though included below, the part of WNTR copied has not affiliation with EPANET

Copyright Notice
=================

Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

License Notice
=================

This software is distributed under the Revised BSD License (see below).
WNTR also leverages a variety of third-party software packages, which
have separate licensing policies.

Revised BSD License
-------------------

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Sandia National Laboratories, nor the names of
  its contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Third-Party Libraries
=================================
WNTR includes source code from the following third-party libraries:

Numpy
-----

Copyright (c) 2005-2019, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

EPANET
------

MIT License

Copyright (c) 2019 (See AUTHORS, https://github.com/OpenWaterAnalytics/EPANET/blob/dev/AUTHORS)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice, list of authors, and this permission notice shall
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



"""

import warnings
import logging
import enum
import time
import numpy as np
import scipy.sparse as sp


warnings.filterwarnings(
    "error", "Matrix is exactly singular", sp.linalg.MatrixRankWarning
)
np.set_printoptions(precision=3, threshold=10000, linewidth=300)



class SolverStatus(enum.IntEnum):
    converged = 1
    error = 0


class NewtonSolver(object):
    """
    Newton Solver class.

    Attributes
    ----------
    log_progress: bool
        If True, the infinity norm of the constraint violation will be logged each iteration
    log_level: int
        The level for logging the infinity norm of the constraint violation
    time_limit: float
        If the wallclock time exceeds time_limit, the newton solver will exit with an error status
    maxiter: int
        If the number of iterations exceeds maxiter, the newton solver will exit with an error status
    tol: float
        The convergence tolerance. If the infinity norm of the constraint violation drops below tol,
        the newton solver will exit with a converged status.
    rho: float
        During the line search, rho is used to reduce the stepsize. It should be strictly between 0 and 1.
    bt_maxiter: int
        The maximum number of line search iterations for each outer iteration
    bt: bool
        If False, a line search will not be used.
    bt_start_iter: int
        A line search will not be used for any iteration prior to bt_start_iter
    """

    def __init__(self, options=None):
        """
        Parameters
        ----------
        options: dict
            A dictionary specifying options for the newton solver. Keys
            should be strings in all caps. See the documentation of the
            NewtonSolver attributes for details on each option. Possible
            keys are:
                | "LOG_PROGRESS" (NewtonSolver.log_progress)
                | "LOG_LEVEL" (NewtonSolver.log_level)
                | "TIME_LIMIT" (NewtonSolver.time_limit)
                | "MAXITER" (NewtonSolver.maxiter)
                | "TOL" (NewtonSolver.tol)
                | "BT_RHO" (NewtonSolver.rho)
                | "BT_MAXITER" (NewtonSolver.bt_maxiter)
                | "BACKTRACKING" (NewtonSolver.bt)
                | "BT_START_ITER" (NewtonSolver.bt_start_iter)
        """
        if options is None:
            options = {}
        self._options = options

        if "LOG_PROGRESS" not in self._options:
            self.log_progress = False
        else:
            self.log_progress = self._options["LOG_PROGRESS"]

        if "LOG_LEVEL" not in self._options:
            self.log_level = logging.DEBUG
        else:
            self.log_level = self._options["LOG_LEVEL"]

        if "TIME_LIMIT" not in self._options:
            self.time_limit = 3600
        else:
            self.time_limit = self._options["TIME_LIMIT"]

        if "MAXITER" not in self._options:
            self.maxiter = 3000
        else:
            self.maxiter = self._options["MAXITER"]

        if "TOL" not in self._options:
            self.tol = 1e-6
        else:
            self.tol = self._options["TOL"]

        if "BT_RHO" not in self._options:
            self.rho = 0.5
        else:
            self.rho = self._options["BT_RHO"]

        if "BT_MAXITER" not in self._options:
            self.bt_maxiter = 100
        else:
            self.bt_maxiter = self._options["BT_MAXITER"]

        if "BACKTRACKING" not in self._options:
            self.bt = True
        else:
            self.bt = self._options["BACKTRACKING"]

        if "BT_START_ITER" not in self._options:
            self.bt_start_iter = 0
        else:
            self.bt_start_iter = self._options["BT_START_ITER"]
            
        self._solution_cost = {}

    def solve(self, model, ostream=None):
        """

        Parameters
        ----------
        model: uh2sc.model.Model

        Returns
        -------
        status: SolverStatus
        message: str
        iter_count: int
        """
        t0 = time.time()

        x = model.get_x()
        if len(x) == 0:
            return (
                SolverStatus.converged,
                "No variables or constraints",
                0,
            )

        use_r_ = False
        new_norm = None
        r_ = None
        
        r_norm_history = np.zeros(self.maxiter)

        # MAIN NEWTON LOOP
        for outer_iter in range(self.maxiter):
            
            if time.time() - t0 >= self.time_limit:
                return (
                    
                    SolverStatus.error,
                    "Time limit exceeded",
                    outer_iter,
                )

            if use_r_:
                r = r_
                r_norm = new_norm
            else:
                r = model.evaluate_residuals()
                
                # do some debugging logging
                model.logging.debug("----------------------------------------")
                model.logging.debug("BEGIN RESIDUALS")
                r_ = r[0:20]
                equ = model.equations_list()[0:20]
                [model.logging.debug(f"residual for {equn_desc}={res})") for res,equn_desc in zip(r_,equ)]
                model.logging.debug("END RESIDUALS")
                model.logging.debug("----------------------------------------")
                
                r_norm = np.max(abs(r))

            if self.log_progress or ostream is not None:
                if outer_iter < self.bt_start_iter:
                    msg = f"iter: {outer_iter:<4d} norm: {r_norm:<10.2e} time: {time.time() - t0:<8.4f}"
                    if self.log_progress:
                        model.logging.debug(msg)
                    if ostream is not None:
                        ostream.write(msg + "\n")
            
            if r_norm < self.tol:
                model.converged_solution_point = True
                model.evaluate_residuals()
                model.logging.info(f"Solution converged with R-norm {r_norm}"
                                   +f" less than solution tolernace {self.tol}")
                model.converged_solution_point = False
                return (
                    SolverStatus.converged,
                    "Solved Successfully",
                    outer_iter,
                )
            
            # detect stagnation in converger solution
            r_norm_history[outer_iter] = r_norm
            if outer_iter > 11:
                last_iters = r_norm_history[outer_iter - 9:outer_iter+1]
                delta = np.max(last_iters) - np.min(last_iters)
                reference = np.abs(np.mean(last_iters)) + 1e-12
                percent_change = (delta / reference) * 100
                if percent_change < 0.3:
                    model.logging.info("Solution stagnated and needs a smaller time step!")

                    return (
                        SolverStatus.error,
                        "Solver stagnated and needs a smaller time step",
                        outer_iter,
                        )
            
            # detect unlikely to converge situations
            if not self._is_likely_to_converge_exponential(r_norm_history, 
                                                           self.maxiter, 
                                                           self.tol, 
                                                           outer_iter):
                fail_str =("Solver is unlikely to converge based on "
                           +"exponential projection, trying a smaller time step")

                model.logging.info("  " + fail_str)
                return (
                    SolverStatus.error,
                    fail_str,
                    outer_iter,
                    )
            
            
            
            J = model.evaluate_jacobian(x)

            # Call Linear solver
            try:
                d = -sp.linalg.spsolve(J, r, permc_spec="COLAMD", use_umfpack=False)
            except sp.linalg.MatrixRankWarning:
                return (
                    SolverStatus.error,
                    "Jacobian is singular at iteration " + str(outer_iter),
                    outer_iter,
                )

            # Backtracking
            alpha = 1.0
            if self.bt and outer_iter >= self.bt_start_iter:
                use_r_ = True
                for iter_bt in range(self.bt_maxiter):
                    x_ = x + alpha * d
                    
                    try:
                        model.load_var_values_from_x(x_)
                        r_ = model.evaluate_residuals()
                    except:
                        alpha = alpha * self.rho
                        continue
                    
                    new_norm = np.max(abs(r_))
                    if new_norm < (1.0 - 0.0001 * alpha) * r_norm:
                        x = x_
                        break
                    else:
                        alpha = alpha * self.rho

                if iter_bt + 1 >= self.bt_maxiter:
                    return (
                        SolverStatus.error,
                        "Line search failed at iteration " + str(outer_iter),
                        outer_iter,
                    )
                if self.log_progress or ostream is not None:
                    msg = f"iter: {outer_iter:<4d} norm: {new_norm:<10.2e} alpha: {alpha:<10.2e} time: {time.time() - t0:<8.4f}"
                    if self.log_progress:
                        model.logging.debug(msg)
                    if ostream is not None:
                        ostream.write(msg + "\n")
            else:
                x += d
                model.load_var_values_from_x(x)
        return (
            SolverStatus.error,
            "Reached maximum number of iterations: " + str(outer_iter),
            outer_iter,
        )
    
    # originally written by AI
    def _is_likely_to_converge_exponential(self,r_norm_history, max_iter, rnorm_tol, iter_num):
        """
        Predict whether exponential convergence will bring r_norm below rnorm_tol
        within the remaining iterations.
    
        Parameters:
        -----------
        r_norm_history : np.ndarray
            1-D array of residual norms
        max_iter : int
            Maximum number of iterations
        rnorm_tol : float
            Convergence threshold
        iter_num : int
            Current iteration number
    
        Returns:
        --------
        likely_to_converge : bool
            True if exponential fit predicts convergence below rnorm_tol
            by max_iter, False otherwise
        """
        
        # Make sure we have at least 10 values
        if iter_num < 10:
            return True  # Not enough history to judge
        
        # Get last 10 r_norm values (from iter_num-10 to iter_num-1)
        y = r_norm_history[iter_num - 10:iter_num]
    
        # Corresponding x values: [-10, -9, ..., -1]
        x = np.arange(-10, 0)
    
        # Model: y = c * exp(a * x) ⇒ log(y) = log(c) + a * x
        log_y = np.log(y)
    
        # Design matrix for linear least squares: [1, x]
        A = np.vstack([np.ones_like(x), x]).T
    
        # Solve for [log(c), a] using least squares
        coeffs, *_ = np.linalg.lstsq(A, log_y, rcond=None)
        log_c, a = coeffs
    
        # Predict r_norm at future iteration: y = c * exp(a * x)
        c = np.exp(log_c)
        steps_remaining = max_iter - iter_num
        predicted_rnorm = c * np.exp(a * steps_remaining)
    
        return predicted_rnorm < rnorm_tol
