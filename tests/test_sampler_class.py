import bilby
import pytest
from pymc_bilby import Pymc


@pytest.fixture()
def SamplerClass():
    return Pymc


@pytest.fixture()
def create_sampler(SamplerClass, bilby_gaussian_likelihood_and_priors, tmp_path):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    def create_fn(**kwargs):
        return SamplerClass(
            likelihood,
            priors,
            outdir=tmp_path / "outdir",
            label="test",
            use_ratio=False,
            **kwargs,
        )

    return create_fn


@pytest.fixture
def sampler(create_sampler):
    return create_sampler()


@pytest.fixture
def expected_default_kwargs():
    return dict(
        draws=500,
        step=None,
        init="auto",
        n_init=200000,
        initvals=None,
        trace=None,
        chains=2,
        cores=1,
        tune=500,
        progressbar=True,
        model=None,
        nuts_kwargs=None,
        step_kwargs=None,
        random_seed=None,
        discard_tuned_samples=True,
        compute_convergence_checks=True,
    )


def test_default_kwargs(sampler, expected_default_kwargs):
    kwargs = expected_default_kwargs.copy()
    kwargs.update(sampler.default_nuts_kwargs)
    kwargs.update(sampler.default_step_kwargs)
    assert sampler.kwargs == kwargs


@pytest.mark.parametrize(
    "equiv", bilby.core.sampler.base_sampler.MCMCSampler.npool_equiv_kwargs
)
def test_translate_kwargs_npool(equiv, create_sampler):
    sampler = create_sampler(**{equiv: 2})
    assert sampler.npool == 2
    assert sampler.kwargs["cores"] == 2
