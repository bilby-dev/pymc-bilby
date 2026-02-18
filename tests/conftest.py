import bilby
import numpy as np
import pytest


def model(x, m, c):
    return m * x + c


@pytest.fixture()
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def bilby_gaussian_likelihood_and_priors(rng):
    x = np.linspace(0, 1, 11)
    injection_parameters = dict(m=0.5, c=0.2)
    sigma = 0.1
    y = model(x, **injection_parameters) + rng.normal(0, sigma, len(x))
    likelihood = bilby.likelihood.GaussianLikelihood(x, y, model, sigma)
    priors = dict(
        m=bilby.core.prior.Uniform(0, 10, "m", boundary="periodic"),
        c=bilby.core.prior.Uniform(-2, 2, "c", boundary="reflective"),
    )
    return likelihood, priors


@pytest.fixture(params=[1, 2])
def npool(request):
    return request.param
