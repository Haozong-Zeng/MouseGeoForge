"""Post-hoc normalized-UV error-scale calibration; no coordinate/model updates."""
import math
import torch


def _validate(residual, sigma):
    if residual.ndim < 1 or residual.shape[-1] != 2 or residual.shape[:-1] != sigma.shape:
        raise ValueError('Expected residual [...,2], sigma [...]')
    if sigma.numel() == 0:
        raise ValueError('Empty calibration set')
    if not torch.isfinite(residual).all() or not torch.isfinite(sigma).all() or not (sigma > 0).all():
        raise ValueError('Calibration requires finite residuals and positive scales')


def fit_temperature(residual, sigma):
    """MLE scalar for a 2D isotropic Gaussian on a separate calibration set.

    This changes uncertainty magnitude, NOT the ordering of point reliability.
    Caller must prevent subject/session/frame leakage and evaluate on held-out data.
    """
    _validate(residual, sigma)
    return float(((residual.double()/sigma.double()[...,None]).square().sum(-1).mean()/2).sqrt().clamp_min(1e-6))


def apply_temperature(sigma, temperature):
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Temperature must be finite and positive')
    if not torch.isfinite(sigma).all() or not (sigma > 0).all():
        raise ValueError('Scales must be finite and positive')
    return sigma*temperature


def uncertainty_metrics(residual, sigma):
    _validate(residual, sigma)
    z2=(residual.double()/sigma.double()[...,None]).square().sum(-1)
    e=residual.double().norm(dim=-1).flatten(); order=sigma.flatten().argsort()
    return dict(points=sigma.numel(),nll=float((.5*z2+2*sigma.double().log()).mean()),
        coverage_68=float((z2<=2.278869).double().mean()),
        coverage_95=float((z2<=5.991465).double().mean()),
        mean_sigma=float(sigma.mean()),
        mean_error_normalized=float(e.mean()),
        # A constant baseline cannot rank reliability. Do not interpret arbitrary
        # tie ordering as a selective prediction curve.
        ranking_informative=bool(sigma.max() > sigma.min()),
        risk_coverage={str(f):(float(e[order[:max(1,int(len(order)*f))]].mean())
                              if f == 1. or sigma.max() > sigma.min() else None)
                       for f in (.25,.5,.75,1.)})
