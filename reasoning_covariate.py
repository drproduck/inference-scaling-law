"""Covariate-augmented pass@k estimators using cached prompt embeddings."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import betaln
from torch import nn

from pass_at_k import pass_at_k_rates_with_sample_variance


def _normalize_hidden(hidden: Union[int, Sequence[int]]) -> list[int]:
    if isinstance(hidden, int):
        return [hidden]
    return list(hidden)


def _mlp_layers(
    in_features: int,
    hidden: Union[int, Sequence[int]],
    out_features: int,
    *,
    dropout_p: float = 0.0,
    use_batch_norm: bool = False,
) -> nn.Sequential:
    hids = _normalize_hidden(hidden)
    layers: list[nn.Module] = []
    d_in = in_features
    for h in hids:
        layers.append(nn.Linear(d_in, h))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        d_in = h
    layers.append(nn.Linear(d_in, out_features))
    return nn.Sequential(*layers)


def _zero_init_last_linear(seq: nn.Sequential) -> None:
    last = seq[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("expected final layer to be nn.Linear")
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class SuccessProbNet(nn.Module):
    def __init__(
        self,
        hidden: Union[int, Sequence[int]] = (64, 64),
        in_features: int = 384,
        dropout_p: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.net = _mlp_layers(
            in_features, hidden, 1, dropout_p=dropout_p, use_batch_norm=use_batch_norm
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x))

    def bce_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self.logits(x), y)


class BetaHyperpriorNet(nn.Module):
    def __init__(
        self,
        hidden: Union[int, Sequence[int]] = (64, 64),
        in_features: int = 384,
        eps: float = 1e-6,
        dropout_p: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.mlp = _mlp_layers(
            in_features, hidden, 2, dropout_p=dropout_p, use_batch_norm=use_batch_norm
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        alpha = F.softplus(h[:, 0]) + self.eps
        beta = F.softplus(h[:, 1]) + self.eps
        return alpha, beta


class BetaResidualNet(nn.Module):
    """Log-scale residuals around pooled EB (alpha0, beta0); zero init => no-covariate prior."""

    def __init__(
        self,
        alpha0: float,
        beta0: float,
        hidden: Union[int, Sequence[int]] = (64, 64),
        in_features: int = 384,
        dropout_p: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "log_alpha0", torch.tensor(float(np.log(max(alpha0, 1e-12))), dtype=torch.float64)
        )
        self.register_buffer(
            "log_beta0", torch.tensor(float(np.log(max(beta0, 1e-12))), dtype=torch.float64)
        )
        self.mlp = _mlp_layers(
            in_features, hidden, 2, dropout_p=dropout_p, use_batch_norm=use_batch_norm
        )
        _zero_init_last_linear(self.mlp)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        log_alpha = self.log_alpha0 + h[:, 0]
        log_beta = self.log_beta0 + h[:, 1]
        return torch.exp(log_alpha), torch.exp(log_beta)


def _torch_lbeta(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)


def _kl_beta(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    alpha0: torch.Tensor,
    beta0: torch.Tensor,
) -> torch.Tensor:
    """KL(Beta(alpha, beta) || Beta(alpha0, beta0)) per row."""
    t = alpha + beta
    return (
        _torch_lbeta(alpha0, beta0)
        - _torch_lbeta(alpha, beta)
        + (alpha - alpha0) * (torch.digamma(alpha) - torch.digamma(t))
        + (beta - beta0) * (torch.digamma(beta) - torch.digamma(t))
    )


def _bb_marginal_loglik_rows(
    alpha: torch.Tensor, beta: torch.Tensor, s: torch.Tensor, n: torch.Tensor
) -> torch.Tensor:
    return (
        torch.lgamma(n + 1.0)
        - torch.lgamma(s + 1.0)
        - torch.lgamma(n - s + 1.0)
        + torch.lgamma(alpha + s)
        + torch.lgamma(beta + n - s)
        - torch.lgamma(alpha + beta + n)
        - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))
    )


def pass_at_k_from_beta_hyper(
    alpha: np.ndarray,
    beta: np.ndarray,
    successes: np.ndarray,
    attempts: np.ndarray,
    k_values: np.ndarray,
) -> np.ndarray:
    """Posterior-mean pass@k per problem, averaged over the batch."""
    post_alpha = np.asarray(alpha, dtype=np.float64) + np.asarray(successes, dtype=np.float64)
    post_beta = (
        np.asarray(beta, dtype=np.float64)
        + np.asarray(attempts, dtype=np.float64)
        - np.asarray(successes, dtype=np.float64)
    )
    k_values = np.atleast_1d(k_values).astype(np.float64)
    pa = post_alpha[:, None]
    pb = post_beta[:, None]
    kv = k_values[None, :]
    log_prob_fail = betaln(pa, pb + kv) - betaln(pa, pb)
    psi = 1.0 - np.exp(np.clip(log_prob_fail, -700.0, 0.0))
    return psi.mean(axis=0)


def pass_at_k_from_marginal_p(p: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    k_values = np.atleast_1d(k_values).astype(np.float64)
    return (1.0 - (1.0 - p[:, None]) ** k_values[None, :]).mean(axis=0)


def _train_net_full_batch(
    net: nn.Module,
    X: np.ndarray,
    y: torch.Tensor,
    *,
    loss_fn,
    lr: float,
    weight_decay: float,
    max_steps: int,
    dtype: torch.dtype = torch.float64,
) -> None:
    x_t = torch.as_tensor(X, dtype=dtype)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    for _ in range(max_steps):
        opt.zero_grad(set_to_none=True)
        loss_fn(net, x_t, y).backward()
        opt.step()
    net.eval()


class CovariateSuccessProbPassAtK:
    def __init__(
        self,
        *,
        hidden: Union[int, Sequence[int]] = [64],
        dropout_p: float = 0.1,
        use_batch_norm: bool = True,
        lr: float = 0.01,
        weight_decay: float = 0.1,
        max_steps: int = 1000,
        dtype: torch.dtype = torch.float64,
    ):
        self.hidden = hidden
        self.dropout_p = dropout_p
        self.use_batch_norm = use_batch_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.dtype = dtype
        self.net_: SuccessProbNet | None = None
        self.p_: np.ndarray | None = None

    def fit(self, X: np.ndarray, successes: np.ndarray, attempts: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        successes = np.asarray(successes, dtype=float)
        attempts = np.asarray(attempts, dtype=float)
        if len(successes) != len(attempts) or X.shape[0] != len(successes):
            raise ValueError("X, successes, attempts must align on n_problems")

        rate = np.where(attempts > 0, successes / attempts, 0.0)
        net = SuccessProbNet(
            hidden=self.hidden,
            in_features=X.shape[1],
            dropout_p=self.dropout_p,
            use_batch_norm=self.use_batch_norm,
        ).to(dtype=self.dtype)

        y_t = torch.as_tensor(rate, dtype=self.dtype)

        def loss_fn(model, x_t, y):
            return model.bce_loss(x_t, y)

        _train_net_full_batch(
            net,
            X,
            y_t,
            loss_fn=loss_fn,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_steps=self.max_steps,
            dtype=self.dtype,
        )
        with torch.no_grad():
            p = (
                net(torch.as_tensor(X, dtype=self.dtype))
                .clamp(1e-7, 1.0 - 1e-7)
                .cpu()
                .numpy()
            )
        self.net_ = net
        self.p_ = np.asarray(p, dtype=np.float64)
        return self

    def predict(self, k_values: np.ndarray) -> np.ndarray:
        if self.p_ is None:
            raise ValueError("Call fit() first.")
        return pass_at_k_from_marginal_p(self.p_, k_values)


class CovariateBetaBinomialPassAtK:
    """
    Covariate-dependent Beta prior with partial pooling toward pooled EB.

    Prompt embeddings x map to prior Beta(alpha(x), beta(x)) via log-scale
    residuals around (alpha0, beta0) from BetaBinomialPassAtK. Training minimizes
    Beta-Binomial marginal NLL plus kl_lambda * KL(prior(x) || Beta(alpha0, beta0))
    (VIB-style bottleneck). Zero-init residuals => exact no-covariate start;
    kl_lambda -> inf => pooled prior for every prompt. Pass@k uses conjugate
    posterior alpha(x)+s, beta(x)+n-s.
    """

    def __init__(
        self,
        *,
        hidden: Union[int, Sequence[int]] = [64,64],
        dropout_p: float = 0.1,
        use_batch_norm: bool = True,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        max_steps: int = 1000,
        kl_lambda: float = 1.,
        dtype: torch.dtype = torch.float64,
    ):
        self.hidden = hidden
        self.dropout_p = dropout_p
        self.use_batch_norm = use_batch_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.kl_lambda = kl_lambda
        self.dtype = dtype
        self.net_: BetaResidualNet | None = None
        self.alpha0_: float | None = None
        self.beta0_: float | None = None
        self.alpha_: np.ndarray | None = None
        self.beta_: np.ndarray | None = None

    def fit(self, X: np.ndarray, successes: np.ndarray, attempts: np.ndarray):
        from pass_at_k import BetaBinomialPassAtK

        X = np.asarray(X, dtype=np.float64)
        successes = np.asarray(successes, dtype=float)
        attempts = np.asarray(attempts, dtype=float)
        if len(successes) != len(attempts) or X.shape[0] != len(successes):
            raise ValueError("X, successes, attempts must align on n_problems")

        eb = BetaBinomialPassAtK(verbose=False).fit(successes, attempts)
        self.alpha0_ = float(eb.alpha_)
        self.beta0_ = float(eb.beta_)

        net = BetaResidualNet(
            alpha0=self.alpha0_,
            beta0=self.beta0_,
            hidden=self.hidden,
            in_features=X.shape[1],
            dropout_p=self.dropout_p,
            use_batch_norm=self.use_batch_norm,
        ).to(dtype=self.dtype)

        s_t = torch.as_tensor(successes, dtype=self.dtype)
        n_t = torch.as_tensor(attempts, dtype=self.dtype)
        a0 = torch.as_tensor(self.alpha0_, dtype=self.dtype)
        b0 = torch.as_tensor(self.beta0_, dtype=self.dtype)

        def loss_fn(model, x_t, _y):
            alpha, beta = model(x_t)
            nll = -_bb_marginal_loglik_rows(alpha, beta, s_t, n_t).mean()
            kl = _kl_beta(alpha, beta, a0, b0).mean()
            return nll + self.kl_lambda * kl

        _train_net_full_batch(
            net,
            X,
            s_t,
            loss_fn=loss_fn,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_steps=self.max_steps,
            dtype=self.dtype,
        )
        with torch.no_grad():
            alpha, beta = net(torch.as_tensor(X, dtype=self.dtype))
            self.alpha_ = alpha.cpu().numpy()
            self.beta_ = beta.cpu().numpy()
        self.net_ = net
        return self

    def predict(self, k_values: np.ndarray, successes: np.ndarray, attempts: np.ndarray) -> np.ndarray:
        if self.alpha_ is None or self.beta_ is None:
            raise ValueError("Call fit() first.")
        return pass_at_k_from_beta_hyper(
            self.alpha_, self.beta_, successes, attempts, k_values
        )


def compute_covariate(
    data: np.ndarray,
    embeddings: np.ndarray,
    k_values: np.ndarray,
    budget_per_problem: int,
    *,
    random_state: int = 42,
    include_baselines: bool = True,
) -> dict:
    """Same train/holdout split as compare_reasoning.compute, plus covariate estimators."""
    from pass_at_k import (
        BetaBinomialPassAtK,
        BetaMixtureNPMLEPassAtK,
        BetaSmoothedNPMLEPassAtK,
        CrossFittedBetaBinomialPassAtK,
        DirichletProcessBetaPassAtK,
        KSplicedPassAtK,
        MixtureBinomialPassAtK,
        NPMLEBinomialPassAtK,
        TailStitchedNPMLEPassAtK,
    )

    rng = np.random.RandomState(random_state)
    n_problems, n_samples = data.shape
    b = min(int(budget_per_problem), n_samples // 2)
    if b <= 0:
        raise ValueError(
            f"budget_per_problem must be in [1, {n_samples // 2}]; got {budget_per_problem}"
        )

    data_est = []
    data_mvue = []
    for i in range(n_problems):
        gens = data[i]
        perm = rng.permutation(n_samples)
        gens = gens[perm]
        data_est.append(gens[:b])
        data_mvue.append(gens[b:])
    data_mvue = np.asarray(data_mvue)

    k_values = np.asarray(k_values, dtype=int)
    max_k = data_mvue.shape[1]
    k_values = np.unique(k_values[(k_values >= 1) & (k_values <= max_k)])
    if k_values.size == 0:
        raise ValueError(f"No k values in [1, {max_k}]")

    successes = np.array([gens.sum(dtype=int) for gens in data_est])
    attempts = np.full(n_problems, b, dtype=int)
    X = np.asarray(embeddings, dtype=np.float64)

    mvue_pass_at_k, mvue_pass_at_k_var = pass_at_k_rates_with_sample_variance(
        data_mvue, k_values
    )

    fit: dict = {
        "k_values": k_values,
        "mvue": mvue_pass_at_k,
        "mvue_var": mvue_pass_at_k_var,
    }

    est_sp = CovariateSuccessProbPassAtK()
    est_sp.fit(X, successes, attempts)
    fit["cov_success_prob"] = est_sp.predict(k_values)

    est_bb = CovariateBetaBinomialPassAtK()
    est_bb.fit(X, successes, attempts)
    fit["cov_beta_bb"] = est_bb.predict(k_values, successes, attempts)

    if not include_baselines:
        return fit

    data_bb = {"successes": successes, "attempts": attempts}
    est = BetaBinomialPassAtK(verbose=False)
    est.fit(data_bb["successes"], data_bb["attempts"])
    fit["estimate_posterior"] = est.predict(k_values, method="posterior")

    est_kspliced = KSplicedPassAtK(
        BetaBinomialPassAtK(verbose=False),
        NPMLEBinomialPassAtK(verbose=False, reg_alpha=0.001),
    )
    est_kspliced.fit(data_bb["successes"], data_bb["attempts"])
    fit["kspliced"] = est_kspliced.predict(k_values)

    est_npmle_reg = NPMLEBinomialPassAtK(verbose=False, reg_alpha=0.001)
    est_npmle_reg.fit(data_bb["successes"], data_bb["attempts"])
    fit["npmle_reg"] = est_npmle_reg.predict(k_values)

    # est_beta_smoothed = BetaSmoothedNPMLEPassAtK(verbose=False)
    # est_beta_smoothed.fit(data_bb["successes"], data_bb["attempts"])
    # fit["beta_smoothed"] = est_beta_smoothed.predict(k_values)

    # est_beta_xfit = CrossFittedBetaBinomialPassAtK(verbose=False)
    # est_beta_xfit.fit(data_bb["successes"], data_bb["attempts"])
    # fit["beta_xfit"] = est_beta_xfit.predict(k_values)

    # est_mixture = MixtureBinomialPassAtK(verbose=False, reg_alpha=0.001)
    # est_mixture.fit(data_bb["successes"], data_bb["attempts"])
    # fit["mixture_binomial"] = est_mixture.predict(k_values)

    # est_dp = DirichletProcessBetaPassAtK()
    # est_dp.fit(data_bb["successes"], data_bb["attempts"])
    # fit["dp"] = est_dp.predict(k_values)

    # est_tail = TailStitchedNPMLEPassAtK()
    # est_tail.fit(data_bb["successes"], data_bb["attempts"])
    # fit["tail"] = est_tail.predict(k_values)

    est_beta_mixture_npmle = BetaMixtureNPMLEPassAtK(verbose=False, nu=8.0)
    est_beta_mixture_npmle.fit(data_bb["successes"], data_bb["attempts"])
    fit["beta_mixture"] = est_beta_mixture_npmle.predict(k_values, method="posterior")

    return fit
