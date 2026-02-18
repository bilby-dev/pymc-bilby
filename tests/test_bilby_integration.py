import logging
import threading
import time
import signal

import bilby
import pytest


@pytest.fixture(scope="session")
def sampler():
    return "pymc"


@pytest.fixture(scope="session")
def sampler_kwargs():
    return dict(draws=50, tune=50, n_init=250)


@pytest.fixture
def outdir(tmp_path):
    return tmp_path / "outdir"


@pytest.fixture
def conversion_function():
    def _conversion_function(parameters, likelihood, prior):
        converted = parameters.copy()
        if "derived" not in converted:
            converted["derived"] = converted["m"] * converted["c"]
        return converted

    return _conversion_function


def run_sampler(
    likelihood, priors, outdir, conversion_function, sampler, npool=None, **kwargs
):
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=sampler,
        outdir=str(outdir),
        save="hdf5",
        npool=npool,
        conversion_function=conversion_function,
        **kwargs,
    )
    return result


def test_run_sampler(
    bilby_gaussian_likelihood_and_priors,
    outdir,
    conversion_function,
    npool,
    sampler,
    sampler_kwargs,
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors
    result = run_sampler(
        likelihood,
        priors,
        outdir,
        conversion_function,
        sampler,
        npool=npool,
        **sampler_kwargs,
    )
    assert "derived" in result.posterior


def test_interrupt_sampler(
    bilby_gaussian_likelihood_and_priors,
    outdir,
    conversion_function,
    sampler,
    sampler_kwargs,
    caplog,
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    bilby_logger = logging.getLogger("bilby")
    bilby_logger.addHandler(caplog.handler)

    started = threading.Event()
    calls = 0

    def trigger_signal(delay=0.8):
        if started.wait(timeout=10):
            time.sleep(delay)
            signal.raise_signal(signal.SIGINT)
        else:
            # if we never started, don't hang the test forever
            pytest.fail("Sampler never began likelihood evaluations")

    thread = threading.Thread(target=trigger_signal, daemon=True)
    thread.start()

    original_log_likelihood = likelihood.log_likelihood

    def slow_log_likelihood(parameters=None):
        nonlocal calls
        calls += 1
        # Bilby tests the likelihood before starting sampling
        if calls > 99:
            started.set()
        return original_log_likelihood(parameters)

    likelihood.log_likelihood = slow_log_likelihood

    label = "test_interrupt"

    sampler_kwargs["draws"] = 10000
    sampler_kwargs["n_init"] = 10000
    try:
        with caplog.at_level("INFO", logger="bilby"):
            with pytest.raises((SystemExit, KeyboardInterrupt)) as exc:
                run_sampler(
                    likelihood,
                    priors,
                    outdir,
                    conversion_function,
                    sampler,
                    exit_code=5,
                    resume=True,
                    label=label,
                    **sampler_kwargs,
                )
    finally:
        bilby_logger.removeHandler(caplog.handler)

    if isinstance(exc.value, SystemExit):
        assert exc.value.code == 5

    assert any(
        "Run interrupted by signal" in record.getMessage()
        and "exit on 5" in record.getMessage()
        for record in caplog.records
    )
