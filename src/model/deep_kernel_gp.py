"""
Deep Kernel GP — drop-in replacement for the isotropic/ARD RBF GP ensemble.

Public API
----------
FeatureExtractor          – L2-normalising feedforward network
DeepKernelGP              – gpytorch.models.ExactGP with a learned feature map
train_deep_kernel_gp()    – trains a single DeepKernelGP, returns losses list
build_deep_kernel_ensemble() – builds M models sharing one trained extractor
"""

import copy
import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


# ── Feature extractor ─────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        projected = self.net(x)
        projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)
        return projected


# ── Deep Kernel GP model ──────────────────────────────────────────────────────

class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 input_dim, hidden_dim=32, latent_dim=8):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    # BoTorch compatibility ────────────────────────────────────────────────────

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    def posterior(self, X, observation_noise: bool = False, **_kwargs):
        """BoTorch-compatible posterior.

        Flattens the leading [..., q, d] dims into [N, d] so GPyTorch's
        ExactGP inference handles all test points in one batch.  Works
        correctly for analytic acquisition functions (q=1) and for
        _get_posterior_mean_stddev calls from dro.py.
        """
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        self.eval()
        self.likelihood.eval()
        X_flat = X.reshape(-1, X.shape[-1])
        with gpytorch.settings.fast_pred_var():
            mvn = self.likelihood(self(X_flat)) if observation_noise else self(X_flat)
        return GPyTorchPosterior(mvn)

    # ── GP forward ────────────────────────────────────────────────────────────

    def forward(self, x):
        projected = self.feature_extractor(x)
        mean = self.mean_module(projected)
        covar = self.covar_module(projected)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ── Training helpers ──────────────────────────────────────────────────────────

def train_deep_kernel_gp(model, likelihood, train_x, train_y,
                          n_iter=100, lr=0.01, weight_decay=1e-2):
    """Train a DeepKernelGP with Adam; weight_decay applies to the extractor only.

    Returns
    -------
    model, likelihood, losses : list[float]
        One loss value per iteration (negative MLL).
    """
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {"params": model.feature_extractor.parameters(),
         "lr": lr, "weight_decay": weight_decay},
        {"params": model.covar_module.parameters(), "lr": lr},
        {"params": model.mean_module.parameters(), "lr": lr},
        {"params": likelihood.parameters(), "lr": lr},
    ])

    mll = ExactMarginalLogLikelihood(likelihood, model)
    losses = []

    for _ in range(n_iter):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    return model, likelihood, losses


def _fine_tune_gp_only(model, likelihood, train_x, train_y, n_iter, lr):
    """Fine-tune covar/mean/likelihood only; feature extractor must be frozen."""
    model.train()
    likelihood.train()

    gp_params = (
        list(model.covar_module.parameters())
        + list(model.mean_module.parameters())
        + list(likelihood.parameters())
    )
    optimizer = torch.optim.Adam(gp_params, lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_iter):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


# ── Ensemble builder ──────────────────────────────────────────────────────────

def build_deep_kernel_ensemble(train_x, train_y,
                                M=5, hidden_dim=32, latent_dim=None,
                                n_iter=100, lr=0.01, weight_decay=1e-2,
                                warm_start_models=None,
                                collect_diagnostics: bool = False):
    """Build M DeepKernelGP models sharing a single trained FeatureExtractor.

    Option A (shared extractor):
      a. Train one anchor model for n_iter iterations.
      b. Copy its feature_extractor weights to all M models.
      c. Freeze the extractor in every model.
      d. Initialise each model's lengthscale from linspace(0.1, 3.0, M).
      e. Fine-tune GP params only for n_iter // 2 iterations per model.

    If warm_start_models is provided (list of M (model, likelihood) tuples from
    the previous BO iteration), weights are copied from those models and only
    n_iter // 2 fine-tuning steps are run.

    Parameters
    ----------
    collect_diagnostics : bool
        When True return ``(ensemble, diagnostics)`` instead of just
        ``ensemble``.  ``diagnostics`` contains:
          - ``anchor_final_loss``: final MLL loss of the anchor / model-0 fit.

    Returns
    -------
    list of M (model, likelihood) tuples, both in eval mode.
    If collect_diagnostics is True, returns (list, dict) instead.
    """
    input_dim = train_x.shape[-1]
    if latent_dim is None:
        latent_dim = 2 * input_dim

    device = train_x.device
    dtype = train_x.dtype
    ls_values = torch.linspace(0.1, 3.0, M)

    # ── Warm-start path ───────────────────────────────────────────────────────
    if warm_start_models is not None:
        ensemble = []
        for i in range(M):
            likelihood = GaussianLikelihood().to(device=device, dtype=dtype)
            model = DeepKernelGP(
                train_x, train_y, likelihood,
                input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
            ).to(device=device, dtype=dtype)

            warm_model, warm_lik = warm_start_models[i]
            model.load_state_dict(warm_model.state_dict())
            likelihood.load_state_dict(warm_lik.state_dict())

            for param in model.feature_extractor.parameters():
                param.requires_grad = False

            model, likelihood = _fine_tune_gp_only(
                model, likelihood, train_x, train_y,
                n_iter=n_iter // 2, lr=lr,
            )
            ensemble.append((model, likelihood))

        if collect_diagnostics:
            # Post-fine-tune MLL loss on model-0 as the diagnostic loss value
            m0, l0 = ensemble[0]
            m0.train(); l0.train()
            try:
                _mll = ExactMarginalLogLikelihood(l0, m0)
                with torch.no_grad():
                    _loss = -_mll(m0(train_x), train_y).item()
            except Exception:
                _loss = float("nan")
            finally:
                m0.eval(); l0.eval()
            return ensemble, {"anchor_final_loss": _loss}
        return ensemble

    # ── Fresh training path ───────────────────────────────────────────────────

    # Step a: train anchor model (trains shared feature extractor)
    anchor_lik = GaussianLikelihood().to(device=device, dtype=dtype)
    anchor_model = DeepKernelGP(
        train_x, train_y, anchor_lik,
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
    ).to(device=device, dtype=dtype)

    anchor_model, anchor_lik, _anchor_losses = train_deep_kernel_gp(
        anchor_model, anchor_lik, train_x, train_y,
        n_iter=n_iter, lr=lr, weight_decay=weight_decay,
    )

    # Step b: snapshot the trained extractor
    trained_extractor_state = copy.deepcopy(
        anchor_model.feature_extractor.state_dict()
    )

    # Steps c-e: build M diverse GP heads on top of the frozen extractor
    ensemble = []
    for i in range(M):
        likelihood = GaussianLikelihood().to(device=device, dtype=dtype)
        model = DeepKernelGP(
            train_x, train_y, likelihood,
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
        ).to(device=device, dtype=dtype)

        # Step b: copy shared extractor weights
        model.feature_extractor.load_state_dict(trained_extractor_state)

        # Step c: freeze extractor
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

        # Step d: diverse initial lengthscale in the latent space
        model.covar_module.base_kernel.initialize(lengthscale=ls_values[i].item())

        # Step e: fine-tune GP params only
        model, likelihood = _fine_tune_gp_only(
            model, likelihood, train_x, train_y,
            n_iter=n_iter // 2, lr=lr,
        )
        ensemble.append((model, likelihood))

    if collect_diagnostics:
        return ensemble, {"anchor_final_loss": _anchor_losses[-1]}
    return ensemble
