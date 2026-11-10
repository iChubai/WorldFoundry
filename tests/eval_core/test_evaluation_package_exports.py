"""Primary symbols re-exported from worldfoundry.evals (lazy __getattr__)."""


def test_evaluation_primary_exports_resolve():
    import worldfoundry.evals as ev

    assert callable(ev.execute_evaluate_run)
    assert callable(ev.run_worldfoundry)
    assert callable(ev.run_model_benchmark)
    assert callable(ev.resolve_model_zoo_runner)
