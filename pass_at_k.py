import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom
from scipy.special import betaln

#################################################
# pass at k basics
def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: 1 - C(n-c, k) / C(n, k). Numerically stable via product of ratios."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    # C(n-c,k)/C(n,k) = prod_{i=0}^{k-1} (n-c-i)/(n-i) — avoids overflow for large n,k
    ratio = 1.0
    for i in range(k):
        ratio *= (n - c - i) / (n - i)
    return 1.0 - ratio

def pass_at_k_rates(data: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """data: (n_problems, n_samples) boolean or 0/1. Returns (len(k_values),) mean pass@k."""
    n_samples = data.shape[1]
    n_correct = np.sum(data, axis=1)  # (n_problems,)
    out = np.zeros(len(k_values))
    for i, k in enumerate(k_values):
        out[i] = np.mean([unbiased_pass_at_k(n_samples, int(c), k) for c in n_correct])
    return out

#################################################
# adaptive sampling algorithms
def kazdan_sampling(oracle_data, total_budget):
    """
    Implements Algorithm 1 and the sampling loop of Algorithm 2.
    """
    num_problems, max_samples = oracle_data.shape
    
    successes = np.zeros(num_problems, dtype=int)
    attempts = np.zeros(num_problems, dtype=int)
    
    for _ in range(total_budget):
        # Calculate estimated success rates (handle division by zero for unattempted problems)
        with np.errstate(divide='ignore', invalid='ignore'):
            p_hat = np.where(attempts > 0, successes / attempts, 0.0)
        
        # Select the hardest problems (minimum p_hat) [cite: 257]
        min_p = np.min(p_hat)
        hardest_indices = np.where(p_hat == min_p)[0]
        
        # Break ties uniformly at random [cite: 258]
        chosen_i = np.random.choice(hardest_indices)
        
        # Sample from the oracle
        sample_idx = attempts[chosen_i]
        if sample_idx < max_samples:
            is_success = oracle_data[chosen_i, sample_idx]
            successes[chosen_i] += is_success
            attempts[chosen_i] += 1
        else:
            # Fallback if we exhaust the oracle for a specific problem
            # In a real scenario with 10k max samples, B would need to be very large to hit this
            break 
            
    return {
        'successes': successes,
        'attempts': attempts
    }

def uniform_sampling(oracle_data, per_problem_budget):
    """
    Implements uniform sampling.
    """
    oracle_data = np.asarray(oracle_data)
    num_problems, max_samples = oracle_data.shape

    successes = np.zeros(num_problems, dtype=int)
    attempts = np.zeros(num_problems, dtype=int)

    if per_problem_budget <= 0 or max_samples <= 0:
        return {
            'successes': successes,
            'attempts': attempts,
        }

    # For each problem (row), sample per_problem_budget entries uniformly without replacement.
    # Cap by max_samples to avoid exhausting the oracle.
    sample_size = int(min(per_problem_budget, max_samples))
    for i in range(num_problems):
        idx = np.random.choice(max_samples, size=sample_size, replace=False)
        successes[i] = int(np.sum(oracle_data[i, idx]))
        attempts[i] = sample_size

    return {
        'successes': successes,
        'attempts': attempts,
    }
#################################################
# beta-binomial fitting (scikit-learn style API)
def bootstrap_pass_at_k_ci(
    estimator_factory,
    successes,
    attempts,
    k_values,
    n_bootstraps=200,
    confidence=0.95,
    predict_configs=None,
    random_state=None,
    verbose=True,
):
    """
    Bootstrap confidence interval for pass@k. Works with any estimator implementing
    fit(successes, attempts) and predict(k_values, **kwargs).

    Parameters
    ----------
    estimator_factory : callable
        Returns a fresh estimator instance, e.g. lambda: BetaBinomialPassAtK(verbose=False).
    successes : array-like of shape (n_problems,)
        Success counts per problem.
    attempts : array-like of shape (n_problems,)
        Attempt counts per problem.
    k_values : array-like
        Values of k for which to compute pass@k.
    n_bootstraps : int, default=200
        Number of bootstrap samples.
    confidence : float, default=0.95
        Confidence level (e.g. 0.95 for 95% CI).
    predict_configs : list of dict, default=None
        Each dict is passed as **kwargs to estimator.predict(k_values, **kwargs).
        Use different configs for different predictors (e.g. BetaBinomialPassAtK's
        method="integrated" vs method="plugin"). If None, defaults to [{}].
        Example: [{"method": "integrated"}, {"method": "plugin", "bias_correct": True}]
    random_state : int or None, default=None
        Seed for reproducibility.
    verbose : bool, default=True
        Whether to print when a bootstrap iteration fails.

    Returns
    -------
    If len(predict_configs) == 1: (point, lower, upper) — each ndarray.
    If len(predict_configs) > 1: (point_1, lower_1, upper_1, point_2, lower_2, upper_2, ...)
    """
    successes = np.asarray(successes)
    attempts = np.asarray(attempts)
    k_values = np.asarray(k_values)
    n = len(successes)
    if predict_configs is None:
        predict_configs = [{}]
    elif isinstance(predict_configs, dict):
        predict_configs = [predict_configs]

    rng = np.random.default_rng(random_state)
    alpha_lo = 100 * (1 - confidence) / 2
    alpha_hi = 100 * (1 + confidence) / 2

    n_configs = len(predict_configs)
    all_preds = [np.zeros((n_bootstraps, len(k_values))) for _ in range(n_configs)]

    for i in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        s, a = successes[idx], attempts[idx]
        try:
            est = estimator_factory()
            est.fit(s, a)
            for j, kwargs in enumerate(predict_configs):
                all_preds[j][i] = est.predict(k_values, **kwargs)
        except Exception as e:
            if verbose:
                print(f"Bootstrap {i} failed: {e}")
            for j in range(n_configs):
                all_preds[j][i] = np.nan

    results = []
    for preds in all_preds:
        results.extend((
            np.nanmean(preds, axis=0),
            np.nanpercentile(preds, alpha_lo, axis=0),
            np.nanpercentile(preds, alpha_hi, axis=0),
        ))

    return tuple(results)


class BetaBinomialPassAtK:
    """
    Estimate pass@k from adaptively sampled (successes, attempts) using Beta-Binomial MLE.

    Fits a Beta(alpha, beta) prior over per-problem success rates, then computes
    expected pass@k via integration or plug-in. Use bootstrap_pass_at_k_ci() for CIs.

    Parameters
    ----------
    random_state : int or None, default=None
        Seed for reproducibility of dynamic sampling.
    verbose : bool, default=True
        Whether to print progress messages.

    Attributes
    ----------
    alpha_ : float
        Fitted alpha parameter of the Beta distribution.
    beta_ : float
        Fitted beta parameter of the Beta distribution.
    cov_ : ndarray of shape (2, 2)
        Estimated covariance matrix of the fitted (alpha, beta) parameters.
    n_problems_in_ : int
        Number of problems seen during fit.
    """

    def __init__(self, random_state=None, verbose=True):
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, successes, attempts):
        """
        Fit the Beta-Binomial model via MLE.
        """
        successes = np.asarray(successes, dtype=float)
        attempts = np.asarray(attempts, dtype=float)
        if len(successes) != len(attempts):
            raise ValueError("successes and attempts must have the same length")

        # ---- Robust initialization ----
        # A decent starting point matters a lot when the true prior is sparse
        # (alpha << 1, beta >> 1), which is common for hard datasets.
        with np.errstate(divide="ignore", invalid="ignore"):
            p_hat = np.where(attempts > 0, successes / attempts, np.nan)
        p_hat = p_hat[np.isfinite(p_hat)]

        def _mom_init(ph):
            # Method-of-moments init for Beta on proportions.
            # If variance is degenerate, fall back to (1, 1).
            if ph.size < 2:
                return 1.0, 1.0
            m = float(np.mean(ph))
            v = float(np.var(ph, ddof=1))
            # Clamp to valid region: v < m(1-m)
            m = min(max(m, 1e-6), 1.0 - 1e-6)
            vmax = m * (1.0 - m)
            if not np.isfinite(v) or v <= 0 or v >= vmax:
                return 1.0, 1.0
            t = vmax / v - 1.0
            a = max(m * t, 1e-5)
            b = max((1.0 - m) * t, 1e-5)
            return a, b

        a0_mom, b0_mom = _mom_init(p_hat)

        # ---- Optimize in log-space for stability ----
        # params = (log_alpha, log_beta) => alpha, beta > 0 automatically.
        def nll_log_params(log_params):
            log_alpha, log_beta = log_params
            alpha = np.exp(log_alpha)
            beta = np.exp(log_beta)
            log_lik = betaln(successes + alpha, attempts - successes + beta) - betaln(alpha, beta)
            return -np.sum(log_lik)

        # A couple of restarts to avoid poor local minima / flat regions.
        inits = [
            (np.log(1.0), np.log(1.0)),
            (np.log(a0_mom), np.log(b0_mom)),
            (np.log(max(1e-3, a0_mom)), np.log(max(1e-3, b0_mom))),
        ]

        best = None
        for x0 in inits:
            res = minimize(nll_log_params, x0, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        result = best

        if not result.success and self.verbose:
            print("Warning: MLE optimization failed to converge.")

        self.alpha_ = float(np.exp(result.x[0]))
        self.beta_ = float(np.exp(result.x[1]))
        
        # Extract the inverse Hessian (Covariance matrix) from the optimizer
        if hasattr(result.hess_inv, "todense"):
            self.cov_ = result.hess_inv.todense()
        else:
            self.cov_ = np.asarray(result.hess_inv)
            
        self.n_problems_in_ = len(successes)
        self.successes_ = successes
        self.attempts_ = attempts
        return self

    def predict(self, k_values, method="integrated", bias_correct=False):
        """
        Predict pass@k for given k values.

        For ``method="integrated"``, returns the pass@k curve under the fitted
        global Beta prior (one value per k). For ``plugin`` and ``posterior``,
        returns the mean of per-problem values over the dataset. After the call,
        ``self._psi`` holds per-k values for ``integrated``, or shape
        ``(n_problems, len(k))`` for ``plugin`` / ``posterior``.
        """
        self._check_fitted()
        k_values = np.asarray(k_values, dtype=float)

        if method == "integrated":
            # Target parameter H(alpha, beta): Expected probability of k failures
            def _prob_fail(a, b):
                return np.exp(betaln(a, b + k_values) - betaln(a, b))

            psi_int = _prob_fail(self.alpha_, self.beta_)
            
            self._psi = 1.0 - psi_int

        elif method == "plugin":
            eb_ests = (self.alpha_ + self.successes_) / (self.alpha_ + self.beta_ + self.attempts_)
            psi_plugin = (1.0 - eb_ests)[:, None] ** k_values[None, :]
            
            # if bias_correct and n >= 2:
            #     correction = (k_values * (k_values - 1) / (2 * n)) * p_mean * (1.0 - p_mean) ** (k_values - 1)
            #     psi_bc = np.clip(psi_plugin - correction, 0.0, 1.0) 
            #     return 1.0 - psi_bc

            self._psi = 1.0 - psi_plugin

        elif method == "posterior":
            # 1. Compute posterior Beta parameters for each problem
            post_alpha = self.alpha_ + self.successes_
            post_beta = self.beta_ + self.attempts_ - self.successes_
            
            # 2. Expand dimensions for broadcasting (Num_Problems x Num_K_Values)
            pa = post_alpha[:, None]
            pb = post_beta[:, None]
            k_val = k_values[None, :]
            
            # 3. Compute the expected value of (1-theta)^k under each local posterior
            # E[(1-theta)^k] = B(alpha, beta + k) / B(alpha, beta)
            log_prob_fail = betaln(pa, pb + k_val) - betaln(pa, pb)
            psi_posterior = np.exp(log_prob_fail)

            self._psi = 1.0 - psi_posterior

        else:
            raise ValueError(
                f"method must be 'integrated' or 'plugin' or 'posterior', got {method!r}"
            )

        # plugin/posterior: (n_problems, len_k) — average over problems.
        # integrated: (len_k,) — already the marginal under Beta(alpha_, beta_).
        if self._psi.ndim == 2:
            out = self._psi.mean(axis=0)
        else:
            out = self._psi

        out = np.asarray(out)
        if out.size == 1:
            return float(out.ravel()[0])
        return out

    def predict_posterior(self, k_values, successes, attempts):
        """
        given test (successes, attempts), infer the latent variables, then compute pass@k
        """
        successes = np.asarray(successes, dtype=float)
        attempts = np.asarray(attempts, dtype=float)
        if len(successes) != len(attempts):
            raise ValueError("successes and attempts must have the same length")
        k_values = np.asarray(k_values, dtype=float)
        # 1. Compute posterior Beta parameters for each problem
        post_alpha = self.alpha_ + successes
        post_beta = self.beta_ + attempts - successes
        
        # 2. Expand dimensions for broadcasting (Num_Problems x Num_K_Values)
        pa = post_alpha[:, None]
        pb = post_beta[:, None]
        k_val = k_values[None, :]
        
        # 3. Compute the expected value of (1-theta)^k under each local posterior
        # E[(1-theta)^k] = B(alpha, beta + k) / B(alpha, beta)
        log_prob_fail = betaln(pa, pb + k_val) - betaln(pa, pb)
        psi_posterior = 1.0 - np.exp(log_prob_fail)
        assert psi_posterior.shape == (len(successes), len(k_values))
        return psi_posterior # should be a (Num_Problems, Num_K_Values) array




    def _check_fitted(self):
        if not hasattr(self, "alpha_"):
            raise ValueError("Estimator not fitted. Call fit() first.")

    def get_params(self, deep=True):
        return {"random_state": self.random_state, "verbose": self.verbose}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in ("random_state", "verbose"):
                raise ValueError(f"Invalid parameter {k}")
            setattr(self, k, v)
        return self


class NPMLEBinomialPassAtK:
    """
    Estimate pass@k from adaptively sampled (successes, attempts) using NPMLE.

    Fits a discrete non-parametric prior over a dense grid of per-problem success rates.
    Because of the self-consistency of the EM algorithm, the resulting plug-in 
    expectation over the estimated prior perfectly matches the average of the 
    individual posterior expectations.

    Parameters
    ----------
    m_grid : int, default=300
        The number of uniform grid points to use between 0 and 1.
    max_iter : int, default=5000
        Maximum number of Expectation-Maximization (EM) iterations.
    tol : float, default=1e-6
        Convergence tolerance for the maximum change in grid weights.
    verbose : bool, default=True
        Whether to print convergence messages.
    reg_alpha : float, default=0.0
        Dirichlet regularization strength (pseudo-counts). Values > 0 pull the 
        weights away from absolute sparsity, acting similarly to maximum entropy.
    include_empirical_support : bool, default=True
        If True (default), unique non-zero sample proportions are unioned into the
        grid so observed ``p_hat`` lie on the support. If False, the support is
        only the fixed cubed-spaced baseline of length ``m_grid`` (plus the
        ``epsilon`` floor), so ``len(w_)`` is the same on every ``fit`` call.

    Attributes
    ----------
    t_ : ndarray of shape (n_support_,)
        The discrete grid points representing possible success rates.
    w_ : ndarray of shape (n_support_,)
        The estimated probability mass (weights) assigned to each grid point.
    posterior_weights_ : ndarray of shape (n_problems, n_support_)
        Posterior probability of each grid point given each problem's counts.
    n_support_ : int
        Number of support points ``len(t_)`` after the last ``fit``.
    n_problems_in_ : int
        Number of problems seen during fit.
    """

    def __init__(
        self,
        m_grid=300,
        max_iter=5000,
        tol=1e-6,
        verbose=True,
        reg_alpha=0.0,
        include_empirical_support=True,
    ):
        self.m_grid = m_grid
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.reg_alpha = reg_alpha
        self.include_empirical_support = include_empirical_support


    def fit(self, successes, attempts):
        self.n_problems_in_ = len(successes)
        self.successes_ = successes
        self.attempts_ = attempts
        successes = np.asarray(successes, dtype=float)
        attempts = np.asarray(attempts, dtype=float)
        
        # 1.2 The Information Limit Bound
        epsilon = 1.0 / np.sum(attempts)
        
        # 2. Grid Construction (baseline resolution fixed by constructor m_grid)
        base = np.linspace(0, 1, int(self.m_grid)) ** 3
        self.t_ = epsilon + base * (1.0 - epsilon)

        if self.include_empirical_support:
            with np.errstate(divide="ignore", invalid="ignore"):
                p_hat = np.where(attempts > 0, successes / attempts, 0.0)
            empirical_grid = np.unique(p_hat[p_hat > 0])
            self.t_ = np.unique(np.concatenate([self.t_, empirical_grid]))

        self.n_support_ = len(self.t_)

        # 3. Construct the Likelihood Kernel Manually (Log-Space)
        t_safe = np.clip(self.t_, 1e-10, 1.0 - 1e-10)
        
        y = successes[:, None]
        n = attempts[:, None]
        t_matrix = t_safe[None, :]

        log_L = y * np.log(t_matrix) + (n - y) * np.log(1.0 - t_matrix)
        log_L -= np.max(log_L, axis=1, keepdims=True)
        L = np.exp(log_L)
        L = np.clip(L, 1e-15, None)

        # 4. Expectation-Maximization Loop
        n_sup = self.n_support_
        w = np.ones(n_sup) / n_sup
        for it in range(self.max_iter):
            joint = L * w[None, :]
            
            # Use small epsilon in denominator for safety
            P = joint / (joint.sum(axis=1, keepdims=True) + 1e-20)
            
            # --- APPLY DIRICHLET REGULARIZATION ---
            if self.reg_alpha > 0:
                expected_counts = P.sum(axis=0)
                smoothed_counts = expected_counts + self.reg_alpha
                w_new = smoothed_counts / smoothed_counts.sum()
            else:
                w_new = P.mean(axis=0)
            # --------------------------------------
            
            if np.max(np.abs(w_new - w)) < self.tol:
                if getattr(self, 'verbose', False):
                    print(f"NPMLE converged at iteration {it}")
                break
            w = w_new

        self.w_ = w
        
        final_joint = L * self.w_[None, :]
        final_P = final_joint / (final_joint.sum(axis=1, keepdims=True) + 1e-20)
        self.posterior_weights_ = final_P

        self.posterior_means_ = np.sum(self.t_[None, :] * final_P, axis=1)
        return self

    def predict(self, k_values, method="integrated", bias_correct=False):
        """
        Predict pass@k for given k values.

        After this call, ``self._psi`` holds per-problem pass@k with shape
        ``(n_problems_in_, len(k_values))``. For ``method="integrated"`` every
        row equals the population marginal under ``w_``; for ``"posterior"``
        and ``"plugin"`` rows generally differ.

        Parameters
        ----------
        k_values : array-like
        method : {"integrated", "posterior", "plugin"}, default="integrated"
            ``integrated`` — expectation of pass@k under the global NPMLE mixture.
            ``posterior`` — for each problem, expectation under that problem's
            posterior on the grid; return value averages over problems (and
            matches ``integrated`` at the unregularized NPMLE fixed point).
            ``plugin`` — uses ``posterior_means_`` as in previous versions.
        bias_correct : bool, default=False
            Only used when ``method="plugin"``.
        """
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)
        n_p = self.n_problems_in_

        t_row = np.clip(self.t_[None, :], 1e-10, 1.0 - 1e-10)  # (1, m_grid)
        k_matrix = k_values[:, None]  # (len_k, 1)
        one_minus_t_pow = (1.0 - t_row) ** k_matrix  # (len_k, m_grid)

        if method == "integrated":
            expected_failures = np.sum(self.w_ * one_minus_t_pow, axis=1)
            pass_at_k = 1.0 - expected_failures
            self._psi = np.broadcast_to(pass_at_k, (n_p, len(k_values))).copy()

        elif method == "posterior":
            # E[(1-theta)^k | data_i] = sum_j P_ij (1-t_j)^k
            expected_fail_per_problem = self.posterior_weights_ @ one_minus_t_pow.T
            pass_at_k = 1.0 - np.mean(expected_fail_per_problem, axis=0)
            self._psi = (1.0 - expected_fail_per_problem)

        elif method == "plugin":
            theta_hat = self.posterior_means_[None, :]  # (1, n_problems)

            expected_failures = (1.0 - theta_hat) ** k_matrix  # (len_k, n_problems)

            if bias_correct:
                empirical_means = self.successes_ / self.attempts_
                correction = (
                    k_matrix
                    * (1.0 - theta_hat) ** (k_matrix - 1.0)
                    * (empirical_means[None, :] - theta_hat)
                )
                pass_at_k = np.clip(
                    1.0 - np.mean(expected_failures - correction, axis=1), 0.0, 1.0
                )
                self._psi = np.clip(
                    1.0 - (expected_failures - correction), 0.0, 1.0
                ).T
            else:
                pass_at_k = 1.0 - np.mean(expected_failures, axis=1)
                self._psi = (1.0 - expected_failures).T

        else:
            raise ValueError(
                f"method must be 'integrated', 'posterior', or 'plugin', got {method!r}"
            )

        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k

    def _check_fitted(self):
        if not hasattr(self, "w_"):
            raise ValueError("Estimator not fitted. Call fit() first.")

    def get_params(self, deep=True):
        return {
            "m_grid": self.m_grid,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "verbose": self.verbose,
            "reg_alpha": self.reg_alpha,
            "include_empirical_support": self.include_empirical_support,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in (
                "m_grid",
                "max_iter",
                "tol",
                "verbose",
                "reg_alpha",
                "include_empirical_support",
            ):
                raise ValueError(f"Invalid parameter {k}")
            setattr(self, k, v)
        return self


class BetaMixtureNPMLEPassAtK:
    """
    Estimate pass@k using a Non-parametric Maximum Likelihood Estimator (NPMLE)
    where the prior is a mixture of continuous Beta distributions rather than 
    discrete point masses.

    This prevents the "zero-inflation asymptote" by ensuring all probability 
    components have continuous, non-zero tails, allowing safe extrapolation 
    to large k values.

    Parameters
    ----------
    m_grid : int, default=400
        Number of grid components (Beta distributions) to mix.
    nu : float, default=100.0
        The concentration/smoothing parameter. Higher means sharper spikes 
        (closer to standard NPMLE); lower means wider smoothing.
    max_iter : int, default=5000
        Maximum number of Expectation-Maximization (EM) iterations.
    tol : float, default=1e-6
        Convergence tolerance for the maximum change in component weights.
    verbose : bool, default=False
        Whether to print convergence messages.
    """

    def __init__(self, m_grid=400, nu=100.0, max_iter=5000, tol=1e-6, verbose=False):
        self.m_grid = m_grid
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, successes, attempts):
        successes = np.asarray(successes, dtype=float)
        attempts = np.asarray(attempts, dtype=float)
        self.successes_ = successes
        self.attempts_ = attempts
        if len(successes) != len(attempts):
            raise ValueError("successes and attempts must have the same length")

        # 1. Define the Grid of Means (mu)
        # We use a polynomial grid for high resolution near zero.
        # CRITICAL: We strictly bound mu between 1e-5 and 1-1e-5. 
        # If mu is exactly 0 or 1, the Beta parameters become 0, and betaln returns NaN.
        epsilon = 1e-5
        base = np.linspace(0, 1, self.m_grid) ** 3
        mu_grid = epsilon + base * (1.0 - 2 * epsilon)

        # Inject empirical success rates (safely clipped)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_hat = np.where(attempts > 0, successes / attempts, 0.0)
        empirical_grid = np.unique(p_hat[(p_hat > 0) & (p_hat < 1)])
        
        self.mu_ = np.unique(np.concatenate([mu_grid, empirical_grid]))
        self.m_grid_actual_ = len(self.mu_)

        # 2. Define the Beta parameters for each component
        self.alpha_ = self.mu_ * self.nu
        self.beta_  = (1.0 - self.mu_) * self.nu

        # 3. Compute Beta-Binomial Likelihood Matrix (Log-Space for Stability)
        # L_ij = Beta(y_i + alpha_j, k_i - y_i + beta_j) / Beta(alpha_j, beta_j)
        y = successes[:, None]
        k = attempts[:, None]
        a = self.alpha_[None, :]
        b = self.beta_[None, :]

        log_L = betaln(y + a, k - y + b) - betaln(a, b)
        
        # Log-Sum-Exp Stabilization: Shift each row by its maximum to prevent exp() underflow
        log_L -= np.max(log_L, axis=1, keepdims=True)
        L = np.exp(log_L)
        L = np.clip(L, 1e-15, None)

        # 4. Expectation-Maximization (EM) Loop
        w = np.ones(self.m_grid_actual_) / self.m_grid_actual_

        for it in range(self.max_iter):
            # E-Step: Compute posterior probabilities
            joint = L * w[None, :]
            P = joint / joint.sum(axis=1, keepdims=True)
            
            # M-Step: Update mixture weights
            w_new = P.mean(axis=0)

            if np.max(np.abs(w_new - w)) < self.tol:
                if self.verbose:
                    print(f"Beta-Mixture NPMLE converged at iteration {it}")
                break
            w = w_new
        else:
            if self.verbose:
                print(f"Warning: Reached max_iter ({self.max_iter}) without strict convergence.")

        self.w_ = w
        self.n_problems_in_ = len(successes)
        return self

    def predict(self, k_values, method="integrated"):
        """
        Predict expected pass@k.
        
        Parameters
        ----------
        k_values : array-like
            The values of k to evaluate pass@k for.
        method : {"integrated", "posterior"}, default="integrated"
            "integrated": Evaluates the expectation over the global learned prior.
            "posterior": Evaluates the expectation over the localized posterior 
                         for each individual problem, then averages.
        """
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)
        
        if method == "integrated":
            # 1. Global prior components
            k_matrix = k_values[:, None]  # Shape: (K, 1)
            a = self.alpha_[None, :]      # Shape: (1, M)
            b = self.beta_[None, :]       # Shape: (1, M)

            # 2. Integrate (1-theta)^k against each global Beta component
            log_fail_prob = betaln(a, b + k_matrix) - betaln(a, b)
            fail_prob = np.exp(log_fail_prob)
            
            # 3. Weight by global mixture weights (w_)
            expected_failures = np.sum(self.w_[None, :] * fail_prob, axis=1)
            pass_at_k = 1.0 - expected_failures
            
        elif method == "posterior":
            # 1. Local Posterior Beta Parameters for all problems and components
            y = self.successes_[:, None]      # Shape: (N, 1)
            m = self.attempts_[:, None]       # Shape: (N, 1)
            a = self.alpha_[None, :]          # Shape: (1, M)
            b = self.beta_[None, :]           # Shape: (1, M)
            
            post_a = a + y                    # Shape: (N, M)
            post_b = b + m - y                # Shape: (N, M)
            
            # 2. Calculate Local Posterior Mixture Weights (P_ij)
            # The probability that problem i belongs to component j
            log_L = betaln(post_a, post_b) - betaln(a, b)
            log_L -= np.max(log_L, axis=1, keepdims=True) # Stabilize
            L = np.exp(log_L)
            
            joint = L * self.w_[None, :]
            P = joint / joint.sum(axis=1, keepdims=True)  # Shape: (N, M)
            
            # 3. Expand dimensions for 3D broadcasting (Problems x Components x K_values)
            pa = post_a[:, :, None]           # Shape: (N, M, 1)
            pb = post_b[:, :, None]           # Shape: (N, M, 1)
            k_val = k_values[None, None, :]   # Shape: (1, 1, K)
            
            # 4. Integrate (1-theta)^k against each local posterior component
            log_fail_component = betaln(pa, pb + k_val) - betaln(pa, pb)
            fail_prob_component = np.exp(log_fail_component) # Shape: (N, M, K)
            
            # 5. Weight by local posterior weights (P) and average across problems
            P_expanded = P[:, :, None]        # Shape: (N, M, 1)
            expected_fail_per_problem = np.sum(P_expanded * fail_prob_component, axis=1) # Shape: (N, K)
            pass_at_k = 1.0 - expected_fail_per_problem.mean(axis=0)                       # Shape: (K,)
            
        else:
            raise ValueError(f"method must be 'integrated' or 'posterior', got {method!r}")

        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k

    def _check_fitted(self):
        if not hasattr(self, "w_"):
            raise ValueError("Estimator not fitted. Call fit() first.")

    def get_params(self, deep=True):
        return {
            "m_grid": self.m_grid, 
            "nu": self.nu, 
            "max_iter": self.max_iter, 
            "tol": self.tol, 
            "verbose": self.verbose
        }

    def set_params(self, **params):
        for param, value in params.items():
            if param not in self.get_params():
                raise ValueError(f"Invalid parameter {param}")
            setattr(self, param, value)
        return self


import numpy as np
from scipy.stats import binom
from scipy.interpolate import BSpline
from scipy.optimize import minimize

class EfronGModelPassAtK:
    """
    Empirical Bayes Pass@k Estimator using Efron's g-modeling.
    
    Models the unknown prior density as an exponential family distribution 
    based on a natural cubic B-spline basis. This guarantees a smooth, 
    continuous prior, eliminating boundary bias and discrete overfitting.

    Parameters
    ----------
    m_grid : int, default=400
        Number of grid points to evaluate the splines over.
    df : int, default=5
        Degrees of freedom (number of spline basis functions). Efron typically 
        uses 5. Higher df means more flexibility but higher variance.
    l2_reg : float, default=1e-4
        A tiny Ridge penalty on the spline coefficients to ensure the convex 
        optimizer never diverges.
    verbose : bool, default=False
        Whether to print convergence details.
    """
    def __init__(self, m_grid=400, df=5, l2_reg=1e-4, verbose=False):
        if df < 4:
            raise ValueError("df must be >= 4 for cubic B-splines.")
        self.m_grid = m_grid
        self.df = df
        self.l2_reg = l2_reg
        self.verbose = verbose

    def _get_bspline_basis(self, x):
        """Generates a cubic B-spline design matrix for the grid x."""
        degree = 3
        # Efron distributes knots based on the quantiles of the grid
        n_inner_knots = self.df - degree - 1
        quantiles = np.linspace(0, 1, n_inner_knots + 2)
        knots = np.quantile(x, quantiles)
        
        # Pad knots for boundary conditions
        t_knots = np.concatenate(([knots[0]]*degree, knots, [knots[-1]]*degree))
        
        Q = np.zeros((len(x), self.df))
        for i in range(self.df):
            c = np.zeros(self.df)
            c[i] = 1.0
            Q[:, i] = BSpline(t_knots, c, degree)(x)
        return Q

    def fit(self, successes, attempts):
        successes = np.asarray(successes, dtype=float)
        attempts = np.asarray(attempts, dtype=float)
        N = len(successes)

        # 1. Define Grid (We still use polynomial to maintain high resolution near zero)
        epsilon = 1.0 / np.sum(attempts)
        base = np.linspace(0, 1, self.m_grid) ** 3
        self.t_ = epsilon + base * (1.0 - epsilon)
        
        # 2. Build the Spline Basis Matrix (Q)
        self.Q_ = self._get_bspline_basis(self.t_)

        # 3. Construct Likelihood Matrix
        L = binom.pmf(successes[:, None], attempts[:, None], self.t_[None, :])
        L = np.clip(L, 1e-15, None)

        # 4. Define Objective and Gradient
        def objective_and_gradient(alpha):
            # Calculate weights: w = exp(Q * alpha) / sum(exp(Q * alpha))
            log_w_unnorm = self.Q_ @ alpha
            log_w_unnorm -= np.max(log_w_unnorm)  # Prevent exp overflow
            w_unnorm = np.exp(log_w_unnorm)
            w = w_unnorm / np.sum(w_unnorm)

            # Marginal likelihoods for each observation
            m_i = L @ w
            m_i_safe = np.clip(m_i, 1e-15, None)
            
            # Negative Log-Likelihood + L2 Regularization
            neg_log_lik = -np.sum(np.log(m_i_safe)) + 0.5 * self.l2_reg * np.sum(alpha**2)

            # --- The Elegant Analytical Gradient ---
            # Posterior probability matrix: P(theta_j | y_i)
            P = (L * w[None, :]) / m_i_safe[:, None]
            
            # Expected spline basis under the PRIOR
            E_prior = w @ self.Q_
            
            # Expected spline basis under the POSTERIORS
            N_j = np.sum(P, axis=0)  # Sum of posteriors across all N observations
            E_post = (N_j @ self.Q_) / N
            
            # Gradient is strictly the difference between Prior and Posterior expectations
            grad = -N * (E_post - E_prior) + self.l2_reg * alpha
            
            return neg_log_lik, grad

        # 5. Optimize Spline Coefficients
        # Initialize alpha to zeros (which creates a uniform prior)
        alpha_init = np.zeros(self.df)
        
        res = minimize(
            objective_and_gradient, 
            alpha_init, 
            method="L-BFGS-B", 
            jac=True,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not res.success and self.verbose:
            print(f"Warning: L-BFGS-B failed to converge: {res.message}")
        elif self.verbose:
            print(f"g-model converged in {res.nit} iterations.")

        self.alpha_ = res.x
        
        # Compute the final, smoothed probability weights
        log_w = self.Q_ @ self.alpha_
        w_unnorm = np.exp(log_w - np.max(log_w))
        self.w_ = w_unnorm / np.sum(w_unnorm)
        
        return self

    def predict(self, k_values):
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)
        
        t_matrix = self.t_[None, :]
        k_matrix = k_values[:, None]
        
        expected_failures = np.sum(self.w_ * ((1.0 - t_matrix) ** k_matrix), axis=1)
        pass_at_k = 1.0 - expected_failures
        
        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k

    def _check_fitted(self):
        if not hasattr(self, "w_"):
            raise ValueError("Estimator not fitted. Call fit() first.")