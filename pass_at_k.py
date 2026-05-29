import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom
from scipy.special import betaln
from scipy.special import logsumexp
from sklearn.model_selection import KFold


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

def pass_at_k_rates_with_sample_variance(data: np.ndarray, k_values: np.ndarray):
    """
    data: (n_problems, n_samples) boolean or 0/1. 
    Returns:
        means: (len(k_values),) mean pass@k curve.
        variances: (len(k_values),) variance of the mean pass@k curve.
    """
    n_problems, n_samples = data.shape
    n_correct = np.sum(data, axis=1)  # (n_problems,)
    
    out_means = np.zeros(len(k_values))
    out_vars = np.zeros(len(k_values))
    
    for i, k in enumerate(k_values):
        prompt_scores = [unbiased_pass_at_k(n_samples, int(c), k) for c in n_correct]
        out_means[i] = np.mean(prompt_scores)
        # 3. Compute the variance of the mean (Standard Error squared)
        if n_problems > 1:
            out_vars[i] = np.var(prompt_scores, ddof=1) / n_problems
        else:
            out_vars[i] = 0.0
            
    return out_means, out_vars + 1e-12

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
        # self.t_ = epsilon + base * (1.0 - epsilon)
        self.t_ = base

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


# class BetaMixtureNPMLEPassAtK:
#     """
#     Estimate pass@k using a Mixture of Continuous Beta distributions via EM.

#     Upgrades:
#     - Added `reg_alpha` (Dirichlet Regularization) to the M-Step. This stabilizes
#       the highly collinear overlapping continuous kernels, preventing the EM from 
#       collapsing into noise.
#     - Default `nu` lowered to 8.0 to match the theoretically optimal heavy-tail
#       smoothing for small-N datasets like AIME and HMMT.

#     Parameters
#     ----------
#     m_grid : int, default=400
#         Number of grid components (Beta distributions) to mix.
#     nu : float, default=8.0
#         The concentration/smoothing parameter. Lower means wider smoothing.
#     reg_alpha : float, default=0.001
#         Dirichlet pseudo-counts added to the EM M-step. Crucial for stabilizing
#         the continuous mixture.
#     max_iter : int, default=5000
#         Maximum number of Expectation-Maximization (EM) iterations.
#     tol : float, default=1e-6
#         Convergence tolerance for the maximum change in component weights.
#     verbose : bool, default=False
#         Whether to print convergence messages.
#     """

#     def __init__(self, m_grid=400, nu=8.0, reg_alpha=0.001, max_iter=5000, tol=1e-6, verbose=False):
#         self.m_grid = m_grid
#         self.nu = nu
#         self.reg_alpha = reg_alpha
#         self.max_iter = max_iter
#         self.tol = tol
#         self.verbose = verbose

#     def fit(self, successes, attempts):
#         successes = np.asarray(successes, dtype=float)
#         attempts = np.asarray(attempts, dtype=float)
#         self.successes_ = successes
#         self.attempts_ = attempts
#         self.n_problems_in_ = len(successes)
        
#         if len(successes) != len(attempts):
#             raise ValueError("successes and attempts must have the same length")

#         # 1. Define the Grid of Means (mu)
#         # We strictly bound mu between 1e-5 and 1-1e-5. 
#         epsilon = 1e-5
#         base = np.linspace(0, 1, self.m_grid) ** 3
#         mu_grid = epsilon + base * (1.0 - 2 * epsilon)

#         # Inject empirical success rates (safely clipped)
#         with np.errstate(divide="ignore", invalid="ignore"):
#             p_hat = np.where(attempts > 0, successes / attempts, 0.0)
#         empirical_grid = np.unique(p_hat[(p_hat > 0) & (p_hat < 1)])
        
#         self.mu_ = np.unique(np.concatenate([mu_grid, empirical_grid]))
#         self.m_grid_actual_ = len(self.mu_)

#         # 2. Define the Beta parameters for each component
#         self.alpha_ = self.mu_ * self.nu
#         self.beta_  = (1.0 - self.mu_) * self.nu

#         # 3. Compute Beta-Binomial Likelihood Matrix (Log-Space for Stability)
#         y = successes[:, None]
#         k = attempts[:, None]
#         a = self.alpha_[None, :]
#         b = self.beta_[None, :]

#         log_L = betaln(y + a, k - y + b) - betaln(a, b)
        
#         # Log-Sum-Exp Stabilization
#         log_L -= np.max(log_L, axis=1, keepdims=True)
#         L = np.exp(log_L)
#         L = np.clip(L, 1e-15, None)

#         # 4. Expectation-Maximization (EM) Loop with MAP Estimation
#         w = np.ones(self.m_grid_actual_) / self.m_grid_actual_

#         for it in range(self.max_iter):
#             # E-Step: Compute posterior responsibilities
#             joint = L * w[None, :]
#             P = joint / joint.sum(axis=1, keepdims=True)
            
#             # M-Step: MAP Update with Dirichlet Regularization
#             # Instead of standard MLE (P.mean), we add pseudo-counts
#             sum_resp = P.sum(axis=0)
#             w_new = (sum_resp + self.reg_alpha) / (self.n_problems_in_ + self.m_grid_actual_ * self.reg_alpha)

#             if np.max(np.abs(w_new - w)) < self.tol:
#                 if self.verbose:
#                     print(f"Beta-Mixture NPMLE converged at iteration {it}")
#                 break
#             w = w_new
#         else:
#             if self.verbose:
#                 print(f"Warning: Reached max_iter ({self.max_iter}) without strict convergence.")

#         self.w_ = w
#         return self

#     def predict(self, k_values, method="integrated"):
#         """Predict expected pass@k."""
#         self._check_fitted()
#         k_values = np.atleast_1d(k_values).astype(float)
        
#         if method == "integrated":
#             k_matrix = k_values[:, None]  
#             a = self.alpha_[None, :]     
#             b = self.beta_[None, :]       

#             log_fail_prob = betaln(a, b + k_matrix) - betaln(a, b)
#             fail_prob = np.exp(log_fail_prob)
            
#             expected_failures = np.sum(self.w_[None, :] * fail_prob, axis=1)
#             pass_at_k = 1.0 - expected_failures
            
#         elif method == "posterior":
#             y = self.successes_[:, None]      
#             m = self.attempts_[:, None]       
#             a = self.alpha_[None, :]          
#             b = self.beta_[None, :]           
            
#             post_a = a + y                    
#             post_b = b + m - y                
            
#             log_L = betaln(post_a, post_b) - betaln(a, b)
#             log_L -= np.max(log_L, axis=1, keepdims=True) 
#             L = np.exp(log_L)
            
#             joint = L * self.w_[None, :]
#             P = joint / joint.sum(axis=1, keepdims=True)  
            
#             pa = post_a[:, :, None]           
#             pb = post_b[:, :, None]           
#             k_val = k_values[None, None, :]   
            
#             log_fail_component = betaln(pa, pb + k_val) - betaln(pa, pb)
#             fail_prob_component = np.exp(log_fail_component) 
            
#             P_expanded = P[:, :, None]        
#             expected_fail_per_problem = np.sum(P_expanded * fail_prob_component, axis=1) 
#             pass_at_k = 1.0 - expected_fail_per_problem.mean(axis=0)                       
            
#         else:
#             raise ValueError(f"method must be 'integrated' or 'posterior', got {method!r}")

#         if pass_at_k.size == 1:
#             return float(pass_at_k[0])
#         return pass_at_k

#     def _check_fitted(self):
#         if not hasattr(self, "w_"):
#             raise ValueError("Estimator not fitted. Call fit() first.")


class BetaMixtureNPMLEPassAtK:
    """
    Estimate pass@k using a Mixture of Continuous Beta distributions via EM.

    Upgrades:
    - Added `reg_alpha` (Dirichlet Regularization) to the M-Step. This stabilizes
      the highly collinear overlapping continuous kernels, preventing the EM from 
      collapsing into noise.
    - Autotuned concentration (`nu="auto"`): Estimates the optimal global bandwidth 
      by fitting a global Beta-Binomial MLE prior before initializing the mixture grid.

    Parameters
    ----------
    m_grid : int, default=400
        Number of grid components (Beta distributions) to mix.
    nu : float or str, default="auto"
        The concentration/smoothing parameter. If "auto", it is estimated via 
        a global Beta-Binomial Maximum Likelihood fit on the dataset. Lower means wider smoothing.
    reg_alpha : float, default=0.001
        Dirichlet pseudo-counts added to the EM M-step. Crucial for stabilizing
        the continuous mixture.
    max_iter : int, default=5000
        Maximum number of Expectation-Maximization (EM) iterations.
    tol : float, default=1e-6
        Convergence tolerance for the maximum change in component weights.
    verbose : bool, default=False
        Whether to print convergence messages.
    """

    def __init__(self, m_grid=400, nu="auto", reg_alpha=0.001, max_iter=5000, tol=1e-6, verbose=False):
        self.m_grid = m_grid
        self.nu = nu
        self.reg_alpha = reg_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _estimate_global_nu(self, successes, attempts):
        """
        Estimates the optimal global concentration (nu = alpha + beta) 
        using a robust Beta-Binomial MLE with multi-start optimization.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            p_hat = np.where(attempts > 0, successes / attempts, np.nan)
        p_hat = p_hat[np.isfinite(p_hat)]

        def _mom_init(ph):
            if ph.size < 2:
                return 1.0, 1.0
            m = float(np.mean(ph))
            v = float(np.var(ph, ddof=1))
            m = min(max(m, 1e-6), 1.0 - 1e-6)
            vmax = m * (1.0 - m)
            if not np.isfinite(v) or v <= 0 or v >= vmax:
                return 1.0, 1.0
            t = vmax / v - 1.0
            a = max(m * t, 1e-5)
            b = max((1.0 - m) * t, 1e-5)
            return a, b

        a0_mom, b0_mom = _mom_init(p_hat)

        def nll_log_params(log_params):
            alpha = np.exp(log_params[0])
            beta = np.exp(log_params[1])
            log_lik = betaln(successes + alpha, attempts - successes + beta) - betaln(alpha, beta)
            return -np.sum(log_lik)

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

        if not best.success and self.verbose:
            print("Warning: Global EB MLE optimization failed to converge. Falling back to best found.")

        global_alpha = float(np.exp(best.x[0]))
        global_beta = float(np.exp(best.x[1]))
        return global_alpha + global_beta

    def tune_nu(self, successes, attempts, nu_candidates=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0], n_splits=5):
        """
        Dynamically find the optimal concentration (nu) using K-Fold Cross Validation.
        Minimizes the out-of-fold Negative Log-Likelihood.
        """
        successes = np.asarray(successes, dtype=float)
        attempts = np.asarray(attempts, dtype=float)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        best_nu = None
        best_nll = np.inf
        
        if self.verbose:
            print(f"Starting CV tuning for nu across candidates: {nu_candidates}")

        for test_nu in nu_candidates:
            fold_nlls = []
            
            for train_idx, val_idx in kf.split(successes):
                # 1. Train the mixture weights on the training fold using test_nu
                # We temporarily override self.nu
                self.nu = test_nu
                
                # Fit the model (this sets self.w_, self.mu_, etc.)
                # Note: We suppress verbose output during CV
                prev_verbose = self.verbose
                self.verbose = False
                self.fit(successes[train_idx], attempts[train_idx])
                self.verbose = prev_verbose
                
                # 2. Evaluate Log-Likelihood on the validation fold
                y_val = successes[val_idx][:, None]
                k_val = attempts[val_idx][:, None]
                a = self.alpha_[None, :]
                b = self.beta_[None, :]

                # L_val shape: (N_val, num_kernels)
                log_L_val = betaln(y_val + a, k_val - y_val + b) - betaln(a, b)
                
                # Mixture likelihood: log(sum_j w_j * L_val_{i,j})
                # Use logsumexp for stability
                log_w = np.log(np.clip(self.w_, 1e-15, 1.0))
                
                max_log_L = np.max(log_L_val, axis=1, keepdims=True)
                L_val_scaled = np.exp(log_L_val - max_log_L)
                
                # log(sum(w * L)) = max_log + log(sum(w * L_scaled))
                val_log_lik = max_log_L.squeeze() + np.log(np.sum(self.w_[None, :] * L_val_scaled, axis=1))
                
                # Calculate mean Negative Log-Likelihood for this fold
                fold_nlls.append(-np.mean(val_log_lik))
                
            mean_nll = np.mean(fold_nlls)
            if self.verbose:
                print(f"  nu = {test_nu:5.1f} | Mean Val NLL: {mean_nll:.4f}")
                
            if mean_nll < best_nll:
                best_nll = mean_nll
                best_nu = test_nu

        if self.verbose:
            print(f"Optimal nu found: {best_nu}")
            
        # Set the optimal nu and do a final fit on the entire dataset
        self.nu = best_nu
        self.fit(successes, attempts)
        
        return self

    def fit(self, successes, attempts):
        successes = np.asarray(successes, dtype=float)
        attempts = np.asarray(attempts, dtype=float)
        self.successes_ = successes
        self.attempts_ = attempts
        self.n_problems_in_ = len(successes)
        
        if len(successes) != len(attempts):
            raise ValueError("successes and attempts must have the same length")

        # 1. Establish the Concentration Parameter (nu)
        if self.nu == "auto":
            self.nu_ = self._estimate_global_nu(successes, attempts)
            if self.verbose:
                print(f"Auto-estimated global concentration (nu_): {self.nu_:.4f}")
        else:
            self.nu_ = float(self.nu)

        # 2. Define the Grid of Means (mu)
        # We strictly bound mu between 1e-5 and 1-1e-5. 
        epsilon = 1e-5
        base = np.linspace(0, 1, self.m_grid) ** 3
        mu_grid = epsilon + base * (1.0 - 2 * epsilon)

        # Inject empirical success rates (safely clipped)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_hat = np.where(attempts > 0, successes / attempts, 0.0)
        empirical_grid = np.unique(p_hat[(p_hat > 0) & (p_hat < 1)])
        
        self.mu_ = np.unique(np.concatenate([mu_grid, empirical_grid]))
        self.m_grid_actual_ = len(self.mu_)

        # 3. Define the Beta parameters for each component
        self.alpha_ = self.mu_ * self.nu_
        self.beta_  = (1.0 - self.mu_) * self.nu_

        # 4. Compute Beta-Binomial Likelihood Matrix (Log-Space for Stability)
        y = successes[:, None]
        k = attempts[:, None]
        a = self.alpha_[None, :]
        b = self.beta_[None, :]

        log_L = betaln(y + a, k - y + b) - betaln(a, b)
        
        # Log-Sum-Exp Stabilization
        log_L -= np.max(log_L, axis=1, keepdims=True)
        L = np.exp(log_L)
        L = np.clip(L, 1e-15, None)

        # 5. Expectation-Maximization (EM) Loop with MAP Estimation
        w = np.ones(self.m_grid_actual_) / self.m_grid_actual_

        for it in range(self.max_iter):
            # E-Step: Compute posterior responsibilities
            joint = L * w[None, :]
            P = joint / joint.sum(axis=1, keepdims=True)
            
            # M-Step: MAP Update with Dirichlet Regularization
            sum_resp = P.sum(axis=0)
            w_new = (sum_resp + self.reg_alpha) / (self.n_problems_in_ + self.m_grid_actual_ * self.reg_alpha)

            if np.max(np.abs(w_new - w)) < self.tol:
                if self.verbose:
                    print(f"Beta-Mixture NPMLE converged at iteration {it}")
                break
            w = w_new
        else:
            if self.verbose:
                print(f"Warning: Reached max_iter ({self.max_iter}) without strict convergence.")

        self.w_ = w
        return self

    def predict(self, k_values, method="integrated"):
        """Predict expected pass@k."""
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)
        
        if method == "integrated":
            k_matrix = k_values[:, None]  
            a = self.alpha_[None, :]     
            b = self.beta_[None, :]       

            log_fail_prob = betaln(a, b + k_matrix) - betaln(a, b)
            fail_prob = np.exp(log_fail_prob)
            
            expected_failures = np.sum(self.w_[None, :] * fail_prob, axis=1)
            pass_at_k = 1.0 - expected_failures
            
        elif method == "posterior":
            y = self.successes_[:, None]      
            m = self.attempts_[:, None]       
            a = self.alpha_[None, :]          
            b = self.beta_[None, :]           
            
            post_a = a + y                    
            post_b = b + m - y                
            
            log_L = betaln(post_a, post_b) - betaln(a, b)
            log_L -= np.max(log_L, axis=1, keepdims=True) 
            L = np.exp(log_L)
            
            joint = L * self.w_[None, :]
            P = joint / joint.sum(axis=1, keepdims=True)  
            
            pa = post_a[:, :, None]           
            pb = post_b[:, :, None]           
            k_val = k_values[None, None, :]   
            
            log_fail_component = betaln(pa, pb + k_val) - betaln(pa, pb)
            fail_prob_component = np.exp(log_fail_component) 
            
            P_expanded = P[:, :, None]        
            expected_fail_per_problem = np.sum(P_expanded * fail_prob_component, axis=1) 
            pass_at_k = 1.0 - expected_fail_per_problem.mean(axis=0)                        
            
        else:
            raise ValueError(f"method must be 'integrated' or 'posterior', got {method!r}")

        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k

    def _check_fitted(self):
        if not hasattr(self, "w_"):
            raise ValueError("Estimator not fitted. Call fit() first.")

class SplicedPosteriorPassAtK:
    """
    A hybrid estimator that resolves Beta misspecification at small k 
    and NPMLE zero-collapse at large k via Posterior Splicing.

    - Problems with observed successes (y > 0) use the NPMLE posterior, 
      perfectly capturing empirical modes and structural 'lumps' for small k.
    - Problems with zero successes (y == 0) use the Beta posterior, 
      leveraging its alpha < 1 infinite asymptote to provide rigorous 
      tail-regularization for large k extrapolation.
    """
    def __init__(self, beta_estimator, npmle_estimator):
        self.beta = beta_estimator
        self.npmle = npmle_estimator

    def fit(self, successes, attempts):
        self.successes_ = np.asarray(successes, dtype=float)
        self.attempts_ = np.asarray(attempts, dtype=float)
        self.n_problems_in_ = len(self.successes_)
        
        # Fit both independent models
        self.beta.fit(self.successes_, self.attempts_)
        self.npmle.fit(self.successes_, self.attempts_)
        
        return self

    def predict(self, k_values):
        """
        Predict pass@k using the spliced posteriors.
        Note: Because this relies on problem-level routing, it only supports 
        the 'posterior' expectation method.
        """
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)
        
        # 1. Get the expected failures per problem from the Beta model
        # E[(1-theta)^k] = B(alpha_post, beta_post + k) / B(alpha_post, beta_post)
        pa = self.beta.alpha_ + self.successes_
        pb = self.beta.beta_ + self.attempts_ - self.successes_
        
        from scipy.special import betaln
        log_fail_beta = betaln(pa[:, None], pb[:, None] + k_values[None, :]) - betaln(pa[:, None], pb[:, None])
        failures_beta = np.exp(log_fail_beta) # Shape: (n_problems, len_k)
        
        # 2. Get the expected failures per problem from the NPMLE model
        t_row = np.clip(self.npmle.t_[None, :], 1e-10, 1.0 - 1e-10)
        one_minus_t_pow = (1.0 - t_row) ** k_values[:, None]
        failures_npmle = self.npmle.posterior_weights_ @ one_minus_t_pow.T # Shape: (n_problems, len_k)
        
        # 3. Splice! 
        # If y == 0, use Beta tail. If y > 0, use NPMLE empirical precision.
        mask_zero = (self.successes_ == 0)
        
        expected_failures = np.where(
            mask_zero[:, None], 
            failures_beta, 
            failures_npmle
        )
        
        # Average across the dataset
        pass_at_k = 1.0 - np.mean(expected_failures, axis=0)
        self._psi = 1.0 - expected_failures
        
        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k

    def _check_fitted(self):
        if not hasattr(self, "successes_"):
            raise ValueError("Estimator not fitted. Call fit() first.")


class KSplicedPassAtK:
    """
    A temporal-spliced estimator that optimizes for WRMSE during interpolation (k <= m)
    while controlling extrapolation (k > m) with Beta-based tails.

    - k <= m: Uses the Regularized NPMLE, strictly adhering to the empirical modes.
    - k > m: Depends on ``extrapolation``:
             * ``multiplicative_beta_decay`` (default): for k>m, multiples the NPMLE
               failure rate at an ``anchor_k`` by Beta posterior failure ratios relative
               to the same anchor. By default ``anchor_k = m`` (continuous splice at k=m);
               set ``tail_decay_anchor_k`` to an earlier ``k<m`` if you want the decay
               shape tuned from that point without moving the NPMLE cutoff ``m``.
             * ``beta_posterior``: uses the Beta–Binomial posterior pass@k directly
               (no NPMLE anchor in the tail; ``tail_decay_anchor_k`` is ignored).

    Parameters
    ----------
    beta_estimator : estimator object
        An instantiated BetaBinomialPassAtK model.
    npmle_estimator : estimator object
        An instantiated NPMLEBinomialPassAtK model (ideally with reg_alpha=0.001).
    m_budget : int or float, default=None
        The point k at which to splice the curves. If None, it defaults to the
        maximum number of attempts observed in the training data.
    extrapolation : {"multiplicative_beta_decay", "beta_posterior"}, optional
        How to extrapolate beyond ``m_budget``. Default matches the legacy
        multiplicative Beta-decay splice; ``beta_posterior`` switches the tail to pure
        Beta posterior predictions.
    tail_decay_anchor_k : int or float or None, default=None
        For ``extrapolation="multiplicative_beta_decay"`` only: the k at which NPMLE and
        Beta failure rates anchor the Beta decay ratios. Must satisfy
        ``0 < tail_decay_anchor_k <= m``. If ``None``, uses ``m`` (backward compatible).

    Raises
    ------
    ValueError
        If ``extrapolation`` is not recognized, or anchor k is invalid.

    """
    _EXTRAPOLATION_OPTS = frozenset({"multiplicative_beta_decay", "beta_posterior"})
    _EXTRAPOLATION_ALIAS = {
        "beta": "beta_posterior",
        "decay": "multiplicative_beta_decay",
    }

    def __init__(
        self,
        beta_estimator,
        npmle_estimator,
        m_budget=None,
        extrapolation="multiplicative_beta_decay",
        tail_decay_anchor_k=None,
    ):
        if isinstance(extrapolation, str):
            extrapolation = extrapolation.strip().replace("-", "_")
            extrapolation = self._EXTRAPOLATION_ALIAS.get(extrapolation, extrapolation)
        if extrapolation not in self._EXTRAPOLATION_OPTS:
            raise ValueError(
                f"extrapolation must be one of {sorted(self._EXTRAPOLATION_OPTS)} "
                f"(or aliases {sorted(self._EXTRAPOLATION_ALIAS)}), got {extrapolation!r}"
            )
        self.beta = beta_estimator
        self.npmle = npmle_estimator
        self.m_budget = m_budget
        self.extrapolation = extrapolation
        self.tail_decay_anchor_k = tail_decay_anchor_k

    def fit(self, successes, attempts):
        self.successes_ = np.asarray(successes, dtype=float)
        self.attempts_ = np.asarray(attempts, dtype=float)
        
        # Fit both underlying models
        self.beta.fit(self.successes_, self.attempts_)
        self.npmle.fit(self.successes_, self.attempts_)
        
        # Determine the splicing boundary (m)
        if self.m_budget is None:
            # Default to the max budget observed in the dataset
            self.m_  = np.max(self.attempts_)
        else:
            self.m_ = float(self.m_budget)
            
        return self

    def predict(self, k_values):
        """
        Predict pass@k using the k-spliced curve.
        """
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)
        
        # 1. Get raw predictions from both models for the requested k_values
        # Wrap in np.atleast_1d to handle the underlying models unwrapping size-1 arrays to floats
        pass_npmle_k = np.atleast_1d(self.npmle.predict(k_values))
        pass_beta_k = np.atleast_1d(self.beta.predict(k_values, method="posterior"))
        
        failures_npmle_k = 1.0 - pass_npmle_k
        failures_beta_k = 1.0 - pass_beta_k
        
        # 2. Anchors for multiplicative Beta decay at k = decay_anchor_k (default m_)
        if self.extrapolation == "multiplicative_beta_decay":
            anchor_k = (
                float(self.m_)
                if self.tail_decay_anchor_k is None
                else float(self.tail_decay_anchor_k)
            )
            if not np.isfinite(anchor_k) or anchor_k <= 0:
                raise ValueError(
                    f"tail_decay_anchor_k must be positive (got {anchor_k})"
                )
            if anchor_k > self.m_:
                raise ValueError(
                    f"tail_decay_anchor_k must be <= m={self.m_} (got {anchor_k}); "
                    "use NPMLE cutoff m_budget for the splice, not anchor."
                )

            pass_npmle_anchor = np.atleast_1d(self.npmle.predict([anchor_k]))
            pass_beta_anchor = np.atleast_1d(
                self.beta.predict([anchor_k], method="posterior")
            )
            failures_npmle_anchor = 1.0 - pass_npmle_anchor[0]
            failures_beta_anchor = 1.0 - pass_beta_anchor[0]

        # 3. Splice NPMLE where k <= m and chosen tail beyond m
        le_m = k_values <= self.m_
        final_failures = np.zeros_like(k_values, dtype=float)
        final_failures[le_m] = failures_npmle_k[le_m]

        if self.extrapolation == "multiplicative_beta_decay":
            decay_ratio = failures_beta_k[~le_m] / (failures_beta_anchor + 1e-12)
            final_failures[~le_m] = failures_npmle_anchor * decay_ratio
        else:
            # beta_posterior: pure Beta pass@k for k > m
            final_failures[~le_m] = failures_beta_k[~le_m]

        pass_at_k = 1.0 - final_failures
        
        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k

    def _check_fitted(self):
        if not hasattr(self, "m_"):
            raise ValueError("Estimator not fitted. Call fit() first.")


class TailStitchedNPMLEPassAtK(NPMLEBinomialPassAtK):
    """
    Estimate pass@k using NPMLE with post-hoc Extreme Value Theory (EVT) tail stitching.
    
    This inherits the exact EM fitting process of the standard NPMLE, preserving its 
    ability to perfectly capture structural modes (like deterministic problems). 
    However, during prediction, the lowest discrete point mass (the "impossible" problems) 
    is surgically replaced by a continuous Beta(1, beta_tail) distribution with the 
    exact same mean. This guarantees a closed-form, polynomial decay for extreme 
    large-k extrapolation, eliminating zero-collapse.
    """
    
    def predict(self, k_values, method="integrated", bias_correct=False):
        """
        Predict pass@k for given k values with a stitched continuous left tail.
        """
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)
        n_p = self.n_problems_in_

        # --- 1. Tail Stitching Setup ---
        # The grid (self.t_) is strictly sorted via np.unique during fit()
        t_0 = self.t_[0] 
        w_0 = self.w_[0]
        
        # Calculate the Beta parameter that perfectly preserves the mean of the point mass
        beta_tail = (1.0 - t_0) / t_0
        
        # Closed-form integration of expected failures for the continuous Beta tail: 
        # Integral of (1-theta)^k * Beta(1, beta_tail)
        tail_decay = beta_tail / (k_values + beta_tail) # shape: (len_k,)

        # Isolate the rest of the discrete grid
        t_rest = np.clip(self.t_[1:], 1e-10, 1.0 - 1e-10) # shape: (m_grid - 1,)
        w_rest = self.w_[1:]
        
        k_matrix = k_values[:, None] # shape: (len_k, 1)
        one_minus_t_pow_rest = (1.0 - t_rest[None, :]) ** k_matrix # shape: (len_k, m_grid - 1)

        # --- 2. Predictions ---
        if method == "integrated":
            # Failures from the discrete bulk
            expected_failures_rest = np.sum(w_rest * one_minus_t_pow_rest, axis=1)
            
            # Combine with failures from the continuous tail
            expected_failures = expected_failures_rest + (w_0 * tail_decay)
            
            pass_at_k = 1.0 - expected_failures
            self._psi = np.broadcast_to(pass_at_k, (n_p, len(k_values))).copy()

        elif method == "posterior":
            # Isolate the local posterior responsibility for the lowest point vs the rest
            pw_0 = self.posterior_weights_[:, 0]  # shape: (n_problems,)
            pw_rest = self.posterior_weights_[:, 1:] # shape: (n_problems, m_grid - 1)
            
            # E[(1-theta)^k | data_i] for the bulk
            expected_fail_rest = pw_rest @ one_minus_t_pow_rest.T # (n_problems, len_k)
            
            # E[(1-theta)^k | data_i] for the tail
            expected_fail_tail = pw_0[:, None] * tail_decay[None, :] # (n_problems, len_k)
            
            expected_fail_per_problem = expected_fail_rest + expected_fail_tail
            pass_at_k = 1.0 - np.mean(expected_fail_per_problem, axis=0)
            self._psi = 1.0 - expected_fail_per_problem

        elif method == "plugin":
            # Note: Because we set the mean of the Beta tail to exactly equal t_0, 
            # the local expected value (theta_hat) for each problem is completely unchanged.
            # Therefore, we can safely use the standard NPMLE logic for the plug-in estimator.
            theta_hat = self.posterior_means_[None, :]  # (1, n_problems)
            expected_failures = (1.0 - theta_hat) ** k_matrix  # (len_k, n_problems)

            if bias_correct:
                empirical_means = self.successes_ / self.attempts_
                correction = (
                    k_matrix
                    * (1.0 - theta_hat) ** (k_matrix - 1.0)
                    * (empirical_means[None, :] - theta_hat)
                )
                expected_failures -= correction

            expected_failures = np.clip(expected_failures, 0.0, 1.0)
            pass_at_k = 1.0 - np.mean(expected_failures, axis=1)
            self._psi = (1.0 - expected_failures).T

        else:
            raise ValueError(
                f"method must be 'integrated', 'posterior', or 'plugin', got {method!r}"
            )

        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k


class MixtureBinomialPassAtK:
    """
    Estimate pass@k using a Semi-Parametric Mixture model (Beta + NPMLE).

    Fits a prior that is a mixture of a continuous Beta(alpha, beta) distribution
    and a discrete non-parametric grid. This combines the NPMLE's ability to model 
    complex structural probabilities with the Beta distribution's crucial tail 
    regularization to prevent zero-collapse during large-k extrapolation.

    Parameters
    ----------
    m_grid : int, default=300
        The number of uniform grid points for the discrete component.
    max_iter : int, default=500
        Maximum number of Expectation-Maximization (EM) iterations.
    tol : float, default=1e-5
        Convergence tolerance for the total log-likelihood.
    verbose : bool, default=True
        Whether to print convergence messages.
    reg_alpha : float, default=0.0
        Dirichlet regularization strength for the discrete weights.
    include_empirical_support : bool, default=True
        If True, unique non-zero sample proportions are unioned into the grid.

    Attributes
    ----------
    lambda_ : float
        The mixture weight assigned to the continuous Beta component (0 to 1).
    alpha_ : float
        Fitted alpha parameter of the Beta distribution.
    beta_ : float
        Fitted beta parameter of the Beta distribution.
    t_ : ndarray of shape (n_support_,)
        The discrete grid points representing possible success rates.
    w_ : ndarray of shape (n_support_,)
        The probability mass assigned to each grid point (sums to 1).
    """

    def __init__(
        self,
        m_grid=300,
        max_iter=500,
        tol=1e-5,
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
        self.successes_ = np.asarray(successes, dtype=float)
        self.attempts_ = np.asarray(attempts, dtype=float)
        
        y = self.successes_
        n = self.attempts_

        # 1. Setup Discrete Grid (NPMLE Base)
        epsilon = 1.0 / np.sum(n)
        base = np.linspace(0, 1, int(self.m_grid)) ** 3
        self.t_ = epsilon + base * (1.0 - epsilon)

        if self.include_empirical_support:
            with np.errstate(divide="ignore", invalid="ignore"):
                p_hat = np.where(n > 0, y / n, 0.0)
            empirical_grid = np.unique(p_hat[p_hat > 0])
            self.t_ = np.unique(np.concatenate([self.t_, empirical_grid]))
            
        self.n_support_ = len(self.t_)
        t_safe = np.clip(self.t_, 1e-10, 1.0 - 1e-10)

        # Precompute the fixed discrete log-likelihoods (N_problems x N_support)
        # Note: We drop the binomial coefficient as it cancels out in EM responsibilities
        self._log_L_discrete = y[:, None] * np.log(t_safe)[None, :] + (n - y)[:, None] * np.log(1.0 - t_safe)[None, :]

        # 2. Initialize Parameters
        self.lambda_ = 0.5  # Start with an even mixture
        self.w_ = np.ones(self.n_support_) / self.n_support_
        
        # MOM init for Beta component
        with np.errstate(divide="ignore", invalid="ignore"):
            ph = np.where(n > 0, y / n, np.nan)
        ph = ph[np.isfinite(ph)]
        
        def _mom_init(p_arr):
            if p_arr.size < 2: return 1.0, 1.0
            m, v = float(np.mean(p_arr)), float(np.var(p_arr, ddof=1))
            m = min(max(m, 1e-6), 1.0 - 1e-6)
            vmax = m * (1.0 - m)
            if not np.isfinite(v) or v <= 0 or v >= vmax: return 1.0, 1.0
            t = vmax / v - 1.0
            return max(m * t, 1e-5), max((1.0 - m) * t, 1e-5)

        self.alpha_, self.beta_ = _mom_init(ph)

        # 3. Expectation-Maximization Loop
        prev_ll = -np.inf
        
        for it in range(self.max_iter):
            # --- E-STEP ---
            # Calculate Beta Log-Likelihoods (N_problems,)
            log_L_beta = betaln(y + self.alpha_, n - y + self.beta_) - betaln(self.alpha_, self.beta_)
            
            # Construct Joint Log-Likelihood Matrix (N_problems x (1 + N_support))
            # Column 0 is the Beta component, Columns 1+ are the Discrete points
            col_beta = np.log(self.lambda_ + 1e-15) + log_L_beta
            cols_discrete = np.log(1.0 - self.lambda_ + 1e-15) + np.log(self.w_ + 1e-15)[None, :] + self._log_L_discrete
            
            log_joint = np.column_stack([col_beta, cols_discrete])
            
            # Responsibilities via logsumexp for stability
            log_marginal = logsumexp(log_joint, axis=1) # (N_problems,)
            log_gamma = log_joint - log_marginal[:, None]
            gamma = np.exp(log_gamma) # (N_problems x (1 + N_support))
            
            gamma_beta = gamma[:, 0]
            gamma_discrete = gamma[:, 1:]
            
            current_ll = np.sum(log_marginal)
            
            # Check convergence
            if np.abs(current_ll - prev_ll) < self.tol:
                if self.verbose:
                    print(f"Mixture EM converged at iteration {it} (Total LL: {current_ll:.2f})")
                break
            prev_ll = current_ll
            
            # --- M-STEP ---
            # 1. Update mixture weight
            # self.lambda_ = np.clip(np.mean(gamma_beta), 1e-5, 1.0 - 1e-5)
            self.lambda_ = np.clip(np.mean(gamma_beta), 0.3, 0.9)
            
            # 2. Update discrete weights (with optional Dirichlet regularization)
            expected_counts = np.sum(gamma_discrete, axis=0)
            if self.reg_alpha > 0:
                smoothed = expected_counts + self.reg_alpha
                self.w_ = smoothed / np.sum(smoothed)
            else:
                self.w_ = expected_counts / np.sum(expected_counts)
                
            # 3. Update Beta parameters (Weighted MLE)
            if self.lambda_ > 1e-4:  # Only optimize if Beta component is actually used
                def nll_log_params(log_params):
                    a, b = np.exp(log_params)
                    ll = betaln(y + a, n - y + b) - betaln(a, b)
                    return -np.sum(gamma_beta * ll)
                
                res = minimize(
                    nll_log_params, 
                    x0=(np.log(self.alpha_), np.log(self.beta_)), 
                    method="L-BFGS-B"
                )
                self.alpha_, self.beta_ = np.exp(res.x)

        # Store final posteriors for predictions
        self.posterior_gamma_beta_ = gamma_beta
        self.posterior_gamma_discrete_ = gamma_discrete

        return self

    def predict(self, k_values, method="integrated"):
        """
        Predict pass@k for given k values.
        """
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)
        
        t_row = np.clip(self.t_[None, :], 1e-10, 1.0 - 1e-10)
        k_matrix = k_values[:, None]
        one_minus_t_pow = (1.0 - t_row) ** k_matrix # (len_k, n_support)
        
        if method == "integrated":
            # 1. Expected failure rate under Global Beta
            fail_beta = np.exp(betaln(self.alpha_, self.beta_ + k_values) - betaln(self.alpha_, self.beta_))
            
            # 2. Expected failure rate under Global NPMLE
            fail_discrete = np.sum(self.w_ * one_minus_t_pow, axis=1)
            
            # Mixture
            expected_failures = (self.lambda_ * fail_beta) + ((1.0 - self.lambda_) * fail_discrete)
            pass_at_k = 1.0 - expected_failures
            
            self._psi = np.broadcast_to(pass_at_k, (self.n_problems_in_, len(k_values))).copy()

        elif method == "posterior":
            # For each problem, the posterior is a mixture of the updated Beta and the discrete grid
            pa = self.alpha_ + self.successes_
            pb = self.beta_ + self.attempts_ - self.successes_
            
            # Failure rate under local Beta Posterior: E[(1-theta)^k] = B(a_post, b_post + k) / B(a_post, b_post)
            log_fail_beta = betaln(pa[:, None], pb[:, None] + k_values[None, :]) - betaln(pa[:, None], pb[:, None])
            fail_beta = np.exp(log_fail_beta) # (n_problems, len_k)
            
            # Failure rate under local NPMLE Posterior
            fail_discrete = self.posterior_gamma_discrete_ @ one_minus_t_pow.T # (n_problems, len_k)
            
            # Combine based on problem-specific component responsibilities
            gamma_b = self.posterior_gamma_beta_[:, None]
            expected_failures = (gamma_b * fail_beta) + fail_discrete 
            
            pass_at_k = 1.0 - np.mean(expected_failures, axis=0)
            self._psi = 1.0 - expected_failures

        elif method == "plugin":
            # Expected theta under local Beta
            theta_beta = (self.alpha_ + self.successes_) / (self.alpha_ + self.beta_ + self.attempts_)
            # Expected theta under local NPMLE
            theta_discrete = np.sum(self.posterior_gamma_discrete_ * self.t_[None, :], axis=1)
            
            # Combined Posterior Mean
            theta_hat = (self.posterior_gamma_beta_ * theta_beta) + theta_discrete
            
            expected_failures = (1.0 - theta_hat[None, :]) ** k_matrix # (len_k, n_problems)
            
            pass_at_k = 1.0 - np.mean(expected_failures, axis=1)
            self._psi = (1.0 - expected_failures).T

        else:
            raise ValueError(f"method must be 'integrated', 'posterior', or 'plugin', got {method!r}")

        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k

    def _check_fitted(self):
        if not hasattr(self, "w_"):
            raise ValueError("Estimator not fitted. Call fit() first.")


class DirichletProcessBetaPassAtK:
    """
    Estimate pass@k using a Truncated Dirichlet Process with a Beta Base Measure.
    
    Instead of a fixed grid, this model dynamically learns the locations (theta) 
    and weights (pi) of K discrete clusters. The Beta base measure acts as a 
    prior on the cluster locations, preventing zero-collapse by dragging the 
    locations slightly away from exactly 0 or 1.
    
    Parameters
    ----------
    max_clusters : int, default=50
        The truncation limit (K) for the Dirichlet Process.
    dp_concentration : float, default=1.0
        The gamma parameter of the DP. Higher values encourage more clusters.
    max_iter : int, default=1000
        Maximum EM iterations.
    tol : float, default=1e-5
        Convergence tolerance.
    """
    def __init__(self, max_clusters=50, dp_concentration=1.0, max_iter=1000, tol=1e-5):
        self.max_clusters = max_clusters
        self.dp_concentration = dp_concentration
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, successes, attempts):
        self.n_problems_in_ = len(successes)
        self.successes_ = np.asarray(successes, dtype=float)
        self.attempts_ = np.asarray(attempts, dtype=float)
        
        y = self.successes_
        n = self.attempts_
        
        # 1. Fit the Base Measure (Global Beta) using Method-of-Moments
        with np.errstate(divide="ignore", invalid="ignore"):
            ph = np.where(n > 0, y / n, np.nan)
        ph = ph[np.isfinite(ph)]
        
        def _mom_init(p_arr):
            if p_arr.size < 2: return 1.0, 1.0
            m, v = float(np.mean(p_arr)), float(np.var(p_arr, ddof=1))
            m = np.clip(m, 1e-6, 1.0 - 1e-6)
            vmax = m * (1.0 - m)
            if not np.isfinite(v) or v <= 0 or v >= vmax: return 1.0, 1.0
            t = vmax / v - 1.0
            return max(m * t, 1.0), max((1.0 - m) * t, 1.0)

        # Alpha_0 and Beta_0 are the Base Measure parameters
        self.alpha_0_, self.beta_0_ = _mom_init(ph)
        
        # 2. Initialize DP Clusters
        # Randomly draw initial cluster locations from the Base Measure
        np.random.seed(42) # Optional: for reproducibility
        self.theta_ = np.random.beta(self.alpha_0_, self.beta_0_, size=self.max_clusters)
        self.pi_ = np.ones(self.max_clusters) / self.max_clusters
        
        prev_ll = -np.inf
        
        # 3. Expectation-Maximization Loop
        for it in range(self.max_iter):
            # Clip theta for numerical stability in log
            t_safe = np.clip(self.theta_, 1e-10, 1.0 - 1e-10)
            
            # --- E-STEP ---
            # Calculate log-likelihood of each problem under each cluster
            # Shape: (N_problems, K_clusters)
            log_L = (y[:, None] * np.log(t_safe)[None, :] + 
                     (n - y)[:, None] * np.log(1.0 - t_safe)[None, :])
            
            log_joint = np.log(self.pi_ + 1e-15)[None, :] + log_L
            
            # Logsumexp to get marginals and responsibilities
            log_marginal = logsumexp(log_joint, axis=1)
            gamma = np.exp(log_joint - log_marginal[:, None]) # Shape: (N, K)
            
            current_ll = np.sum(log_marginal)
            if np.abs(current_ll - prev_ll) < self.tol:
                break
            prev_ll = current_ll
            
            # --- M-STEP ---
            # 1. Update Weights (pi) with Dirichlet Prior (DP Concentration)
            expected_counts = np.sum(gamma, axis=0) # (K,)
            smoothed_counts = expected_counts + (self.dp_concentration / self.max_clusters)
            self.pi_ = smoothed_counts / np.sum(smoothed_counts)
            
            # 2. Update Locations (theta) with Beta Base Measure (Posterior Mean)
            # This is where the magic happens: the Beta prior prevents theta from becoming exactly 0 or 1.
            weighted_successes = np.sum(gamma * y[:, None], axis=0) # (K,)
            weighted_attempts = np.sum(gamma * n[:, None], axis=0) # (K,)
            
            self.theta_ = (weighted_successes + self.alpha_0_) / (weighted_attempts + self.alpha_0_ + self.beta_0_)
            
        # Filter out "dead" clusters (weights near zero) to clean up the model
        active = self.pi_ > 1e-4
        self.pi_ = self.pi_[active] / np.sum(self.pi_[active])
        self.theta_ = self.theta_[active]
        self.n_support_ = len(self.theta_)
        
        # Calculate final posteriors for predictions
        t_safe = np.clip(self.theta_, 1e-10, 1.0 - 1e-10)
        log_L = y[:, None] * np.log(t_safe)[None, :] + (n - y)[:, None] * np.log(1.0 - t_safe)[None, :]
        log_joint = np.log(self.pi_)[None, :] + log_L
        self.posterior_weights_ = np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])

        return self

    def predict(self, k_values, method="integrated"):
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)
        
        t_row = self.theta_[None, :]
        k_matrix = k_values[:, None]
        one_minus_t_pow = (1.0 - t_row) ** k_matrix # (len_k, n_support)
        
        if method == "integrated":
            expected_failures = np.sum(self.pi_ * one_minus_t_pow, axis=1)
            pass_at_k = 1.0 - expected_failures
            self._psi = np.broadcast_to(pass_at_k, (self.n_problems_in_, len(k_values))).copy()
            
        elif method == "posterior":
            expected_fail_per_problem = self.posterior_weights_ @ one_minus_t_pow.T
            pass_at_k = 1.0 - np.mean(expected_fail_per_problem, axis=0)
            self._psi = 1.0 - expected_fail_per_problem
            
        elif method == "plugin":
            theta_hat = np.sum(self.posterior_weights_ * self.theta_[None, :], axis=1)
            expected_failures = (1.0 - theta_hat[None, :]) ** k_matrix
            pass_at_k = 1.0 - np.mean(expected_failures, axis=1)
            self._psi = (1.0 - expected_failures).T
            
        else:
            raise ValueError(f"method must be 'integrated', 'posterior', or 'plugin', got {method!r}")

        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k

    def _check_fitted(self):
        if not hasattr(self, "pi_"):
            raise ValueError("Estimator not fitted. Call fit() first.")




class BetaSmoothedNPMLEPassAtK:
    """
    Beta-Smoothed NPMLE (Kernel Density Estimation).

    This model achieves the mathematical ideal by taking the discrete structural 
    grid of the Regularized NPMLE and "melting" every point mass into a continuous 
    Beta kernel. 

    - For problems with y > 0: The Bayesian update naturally tightens the Beta 
      kernels around the empirical data, minimizing WRMSE at small k.
    - For problems with y = 0: The Bayesian update naturally relies on the lowest 
      Beta kernel (where alpha < 1). This natively triggers Karamata's Tauberian 
      theorem, guaranteeing algebraic O(k^-alpha) decay for safe extrapolation 
      without requiring any hardcoded splicing boundaries.

    Parameters
    ----------
    nu : float, default=8.0
        The precision parameter of the Beta kernels. Higher values make the kernels 
        sharper (closer to raw NPMLE). Lower values apply heavier smoothing.
    reg_alpha : float, default=0.001
        The Dirichlet regularization applied to the underlying NPMLE.
    m_grid : int, default=300
        The number of grid points for the underlying NPMLE.
    """
    def __init__(self, nu=8.0, reg_alpha=0.001, m_grid=300, verbose=False):
        self.nu = nu
        self.reg_alpha = reg_alpha
        self.m_grid = m_grid
        self.verbose = verbose
        # Internally instantiate the optimal baseline NPMLE
        self.npmle = NPMLEBinomialPassAtK(
            m_grid=m_grid, 
            reg_alpha=reg_alpha, 
            verbose=verbose
        )

    def fit(self, successes, attempts):
        self.successes_ = np.asarray(successes, dtype=float)
        self.attempts_ = np.asarray(attempts, dtype=float)
        self.n_problems_in_ = len(self.successes_)

        # 1. Fit the underlying NPMLE to get the structural modes
        self.npmle.fit(self.successes_, self.attempts_)

        # 2. Extract the grid points (t) and mixture weights (w)
        self.w_ = self.npmle.w_
        
        # Clip t slightly away from absolute 0 or 1 to ensure valid Beta parameters
        self.t_ = np.clip(self.npmle.t_, 1e-10, 1.0 - 1e-10)

        # 3. Pre-calculate the Prior Beta parameters for each KDE kernel
        # Mean of kernel j is t_j. Precision is nu.
        self.alpha_prior_ = self.nu * self.t_
        self.beta_prior_  = self.nu * (1.0 - self.t_)

        return self

    def predict(self, k_values, method="posterior"):
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)

        # Prior log-weights: shape (1, n_support)
        log_w = np.log(self.w_ + 1e-15)[None, :]

        # Data arrays: shape (n_problems, 1)
        y = self.successes_[:, None]
        n = self.attempts_[:, None]

        # Prior parameter arrays: shape (1, n_support)
        a = self.alpha_prior_[None, :]
        b = self.beta_prior_[None, :]

        if method == "integrated":
            # Global Failure Rate: E[(1-theta)^k] under the prior mixture
            # B(a, b+k) / B(a, b)
            log_fail_prior = betaln(a, b + k_values[:, None]) - betaln(a, b)
            expected_failures_per_kernel = np.exp(log_fail_prior)  # (len_k, n_support)

            # Weighted sum over kernels
            expected_failures = np.sum(self.w_[None, :] * expected_failures_per_kernel, axis=1)
            
            pass_at_k = 1.0 - expected_failures
            self._psi = np.broadcast_to(pass_at_k, (self.n_problems_in_, len(k_values))).copy()

        elif method in ["posterior", "plugin"]:
            # --- Exact Bayesian Update for the Mixture ---
            
            # 1. Update the parameters of every kernel for every problem
            a_post = a + y  # (n_problems, n_support)
            b_post = b + n - y
            
            # 2. Compute Log Marginal Likelihood of the data under each kernel
            # This determines which kernels best explain which problems
            log_L = betaln(a_post, b_post) - betaln(a, b)
            
            # 3. Calculate Exact Posterior Mixture Weights (Responsibilities)
            log_joint = log_w + log_L
            log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
            gamma = np.exp(log_joint - log_marginal)  # (n_problems, n_support)

            if method == "posterior":
                # E[(1-theta)^k | data] = sum_j gamma_ij * [B(a_post, b_post + k) / B(a_post, b_post)]
                
                # Expand to 3D for broadcasting against k_values
                a_post_3d = a_post[:, :, None]  # (n_problems, n_support, 1)
                b_post_3d = b_post[:, :, None]
                k_3d = k_values[None, None, :]  # (1, 1, len_k)

                log_fail_post = betaln(a_post_3d, b_post_3d + k_3d) - betaln(a_post_3d, b_post_3d)
                fail_post = np.exp(log_fail_post)  # (n_problems, n_support, len_k)

                # Multiply by posterior responsibilities and sum over kernels
                expected_fail_per_problem = np.sum(gamma[:, :, None] * fail_post, axis=1) 
                
                pass_at_k = 1.0 - np.mean(expected_fail_per_problem, axis=0)
                self._psi = 1.0 - expected_fail_per_problem

            elif method == "plugin":
                # Posterior mean of each kernel
                mean_kernel = a_post / (a_post + b_post)
                
                # Overall posterior mean for problem i
                theta_hat = np.sum(gamma * mean_kernel, axis=1)  # (n_problems,)

                expected_failures = (1.0 - theta_hat[None, :]) ** k_values[:, None]
                
                pass_at_k = 1.0 - np.mean(expected_failures, axis=1)
                self._psi = (1.0 - expected_failures).T

        else:
            raise ValueError(f"method must be 'integrated', 'posterior', or 'plugin', got {method!r}")

        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k

    def _check_fitted(self):
        if not hasattr(self, "w_"):
            raise ValueError("Estimator not fitted. Call fit() first.")





# class BetaSmoothedNPMLEPassAtK:
#     """
#     Beta-Smoothed NPMLE (Kernel Density Estimation) with Adaptive Bandwidths.

#     This model achieves the mathematical ideal by taking the discrete structural 
#     grid of the Regularized NPMLE and "melting" every point mass into a continuous 
#     Beta kernel. 

#     Upgrades:
#     - nu="auto": Automatically scales global precision based on dataset size N, 
#       preventing over-smoothing on dense datasets (GPQA) while heavily smoothing 
#       sparse datasets (AIME).
#     - adaptive_bandwidth: Localizes precision across the grid. Kernels in the 
#       noisy center widen to smooth variance, while edge kernels sharpen to 
#       guarantee extreme-value tail asymptotics.

#     Parameters
#     ----------
#     nu : float or "auto", default="auto"
#         The base precision parameter of the Beta kernels. If "auto", it dynamically
#         calculates precision based on the number of observed problems (sqrt(N)).
#     adaptive_bandwidth : bool, default=True
#         Whether to scale precision locally based on grid variance t(1-t).
#     reg_alpha : float, default=0.001
#         The Dirichlet regularization applied to the underlying NPMLE.
#     m_grid : int, default=300
#         The number of grid points for the underlying NPMLE.
#     """
#     def __init__(self, nu="auto", adaptive_bandwidth=True, reg_alpha=0.001, m_grid=300, verbose=False):
#         self.nu = nu
#         self.adaptive_bandwidth = adaptive_bandwidth
#         self.reg_alpha = reg_alpha
#         self.m_grid = m_grid
#         self.verbose = verbose
#         # Internally instantiate the optimal baseline NPMLE
#         self.npmle = NPMLEBinomialPassAtK(
#             m_grid=m_grid, 
#             reg_alpha=reg_alpha, 
#             verbose=verbose
#         )

#     def fit(self, successes, attempts):
#         self.successes_ = np.asarray(successes, dtype=float)
#         self.attempts_ = np.asarray(attempts, dtype=float)
#         self.n_problems_in_ = len(self.successes_)

#         # 1. Fit the underlying NPMLE to get the structural modes
#         self.npmle.fit(self.successes_, self.attempts_)

#         # 2. Extract the grid points (t) and mixture weights (w)
#         self.w_ = self.npmle.w_
        
#         # Clip t slightly away from absolute 0 or 1 to ensure valid Beta parameters
#         self.t_ = np.clip(self.npmle.t_, 1e-10, 1.0 - 1e-10)

#         # 3. Calculate Base Precision
#         if self.nu == "auto":
#             # KDE precision should grow as N grows to prevent over-smoothing dense data.
#             # O(sqrt(N)) is a highly stable heuristic for Beta KDE bandwidths.
#             base_nu = float(np.sqrt(self.n_problems_in_))
#         else:
#             base_nu = float(self.nu)

#         # 4. Calculate Final Adaptive Precision per Kernel
#         if self.adaptive_bandwidth:
#             # Deterministic variance scaling: precision = base_nu / (variance + epsilon)
#             # Epsilon = 0.1 prevents infinite precision (Dirac collapse) exactly at 0 or 1,
#             # guaranteeing that the lowest kernel maintains alpha < 1.
#             epsilon = 0.1
#             self.nu_j_ = base_nu / (self.t_ * (1.0 - self.t_) + epsilon)
#         else:
#             self.nu_j_ = np.full_like(self.t_, base_nu)

#         # 5. Pre-calculate the Prior Beta parameters for each KDE kernel
#         self.alpha_prior_ = self.nu_j_ * self.t_
#         self.beta_prior_  = self.nu_j_ * (1.0 - self.t_)

#         return self

#     def predict(self, k_values, method="posterior"):
#         self._check_fitted()
#         k_values = np.atleast_1d(k_values).astype(float)

#         # Prior log-weights: shape (1, n_support)
#         log_w = np.log(self.w_ + 1e-15)[None, :]

#         # Data arrays: shape (n_problems, 1)
#         y = self.successes_[:, None]
#         n = self.attempts_[:, None]

#         # Prior parameter arrays: shape (1, n_support)
#         # Broadcasting works perfectly with adaptive bandwidths since it's an array
#         a = self.alpha_prior_[None, :]
#         b = self.beta_prior_[None, :]

#         if method == "integrated":
#             # Global Failure Rate: E[(1-theta)^k] under the prior mixture
#             log_fail_prior = betaln(a, b + k_values[:, None]) - betaln(a, b)
#             expected_failures_per_kernel = np.exp(log_fail_prior)  # (len_k, n_support)

#             # Weighted sum over kernels
#             expected_failures = np.sum(self.w_[None, :] * expected_failures_per_kernel, axis=1)
            
#             pass_at_k = 1.0 - expected_failures
#             self._psi = np.broadcast_to(pass_at_k, (self.n_problems_in_, len(k_values))).copy()

#         elif method in ["posterior", "plugin"]:
#             # --- Exact Bayesian Update for the Mixture ---
            
#             # 1. Update the parameters of every kernel for every problem
#             a_post = a + y  # (n_problems, n_support)
#             b_post = b + n - y
            
#             # 2. Compute Log Marginal Likelihood of the data under each kernel
#             log_L = betaln(a_post, b_post) - betaln(a, b)
            
#             # 3. Calculate Exact Posterior Mixture Weights (Responsibilities)
#             log_joint = log_w + log_L
#             log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
#             gamma = np.exp(log_joint - log_marginal)  # (n_problems, n_support)

#             if method == "posterior":
#                 # Expand to 3D for broadcasting against k_values
#                 a_post_3d = a_post[:, :, None]  # (n_problems, n_support, 1)
#                 b_post_3d = b_post[:, :, None]
#                 k_3d = k_values[None, None, :]  # (1, 1, len_k)

#                 log_fail_post = betaln(a_post_3d, b_post_3d + k_3d) - betaln(a_post_3d, b_post_3d)
#                 fail_post = np.exp(log_fail_post)  # (n_problems, n_support, len_k)

#                 # Multiply by posterior responsibilities and sum over kernels
#                 expected_fail_per_problem = np.sum(gamma[:, :, None] * fail_post, axis=1) 
                
#                 pass_at_k = 1.0 - np.mean(expected_fail_per_problem, axis=0)
#                 self._psi = 1.0 - expected_fail_per_problem

#             elif method == "plugin":
#                 mean_kernel = a_post / (a_post + b_post)
#                 theta_hat = np.sum(gamma * mean_kernel, axis=1)  # (n_problems,)
#                 expected_failures = (1.0 - theta_hat[None, :]) ** k_values[:, None]
                
#                 pass_at_k = 1.0 - np.mean(expected_failures, axis=1)
#                 self._psi = (1.0 - expected_failures).T

#         else:
#             raise ValueError(f"method must be 'integrated', 'posterior', or 'plugin', got {method!r}")

#         if pass_at_k.size == 1:
#             return float(pass_at_k[0])
#         return pass_at_k

#     def _check_fitted(self):
#         if not hasattr(self, "w_"):
#             raise ValueError("Estimator not fitted. Call fit() first.")



class CrossFittedBetaBinomialPassAtK:
    """
    Leave-One-Out Cross-Fitted Beta-Binomial Estimator.

    Instead of fitting a single global prior and using it to predict all problems,
    this estimator fits a Beta(alpha, beta) prior on N-1 problems, and uses that
    isolated prior to compute the posterior prediction for the single left-out problem.
    It repeats this for all N problems and averages the results.

    This prevents "double-dipping" (using the same data to shape the prior and 
    update the posterior), significantly reducing overfitting at small sample sizes.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def _fit_mle(self, successes, attempts, warm_start=None):
        """Helper method to fit the Beta-Binomial MLE on a subset of data."""
        with np.errstate(divide="ignore", invalid="ignore"):
            p_hat = np.where(attempts > 0, successes / attempts, np.nan)
        p_hat = p_hat[np.isfinite(p_hat)]

        def _mom_init(ph):
            if ph.size < 2: return 1.0, 1.0
            m = float(np.mean(ph))
            v = float(np.var(ph, ddof=1))
            m = min(max(m, 1e-6), 1.0 - 1e-6)
            vmax = m * (1.0 - m)
            if not np.isfinite(v) or v <= 0 or v >= vmax: return 1.0, 1.0
            t = vmax / v - 1.0
            a = max(m * t, 1e-5)
            b = max((1.0 - m) * t, 1e-5)
            return a, b

        a0_mom, b0_mom = _mom_init(p_hat)

        def nll_log_params(log_params):
            alpha = np.exp(log_params[0])
            beta = np.exp(log_params[1])
            log_lik = betaln(successes + alpha, attempts - successes + beta) - betaln(alpha, beta)
            return -np.sum(log_lik)

        # Use warm start if available to drastically speed up the N LOO iterations
        inits = [(np.log(a0_mom), np.log(b0_mom))]
        if warm_start is not None:
            inits.insert(0, (np.log(warm_start[0]), np.log(warm_start[1])))
        else:
            inits.append((np.log(1.0), np.log(1.0)))

        best = None
        for x0 in inits:
            res = minimize(nll_log_params, x0, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res

        return float(np.exp(best.x[0])), float(np.exp(best.x[1]))

    def fit(self, successes, attempts):
        """
        Fit the N cross-fitted priors.
        """
        self.successes_ = np.asarray(successes, dtype=float)
        self.attempts_ = np.asarray(attempts, dtype=float)
        n = len(self.successes_)
        
        if n < 2:
            raise ValueError("Need at least 2 problems for cross-fitting.")

        # 1. Fit global MLE to use as a warm start for the LOO fits
        global_alpha, global_beta = self._fit_mle(self.successes_, self.attempts_)

        self.loo_alphas_ = np.zeros(n)
        self.loo_betas_ = np.zeros(n)

        # 2. Leave-One-Out Cross-Fitting Loop
        for i in range(n):
            # Create a mask that is True everywhere except index i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            
            loo_succ = self.successes_[mask]
            loo_att = self.attempts_[mask]
            
            # Fit MLE on N-1 problems
            a_i, b_i = self._fit_mle(loo_succ, loo_att, warm_start=(global_alpha, global_beta))
            
            self.loo_alphas_[i] = a_i
            self.loo_betas_[i] = b_i
            
        return self

    def predict(self, k_values):
        """
        Predict pass@k by applying the cross-fitted priors to their respective 
        left-out problems, then averaging the posteriors.
        """
        self._check_fitted()
        k_values = np.atleast_1d(k_values).astype(float)

        # 1. Compute posterior Beta parameters for each problem using its isolated LOO prior
        post_alpha = self.loo_alphas_ + self.successes_
        post_beta = self.loo_betas_ + self.attempts_ - self.successes_
        
        # 2. Expand dimensions for broadcasting (Num_Problems x Num_K_Values)
        pa = post_alpha[:, None]
        pb = post_beta[:, None]
        k_val = k_values[None, :]
        
        # 3. Compute the expected value of (1-theta)^k under each local posterior
        # E[(1-theta)^k] = B(alpha, beta + k) / B(alpha, beta)
        log_prob_fail = betaln(pa, pb + k_val) - betaln(pa, pb)
        expected_fail_per_problem = np.exp(log_prob_fail)

        # Average the expected failures across all cross-fitted problems
        pass_at_k = 1.0 - expected_fail_per_problem.mean(axis=0)

        if pass_at_k.size == 1:
            return float(pass_at_k[0])
        return pass_at_k

    def _check_fitted(self):
        if not hasattr(self, "loo_alphas_"):
            raise ValueError("Estimator not fitted. Call fit() first.")