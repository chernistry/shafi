from shafi import eval as eval_mod


def test_eval_module_lazy_exports_resolve() -> None:
    assert eval_mod.load_golden_dataset is not None
    assert eval_mod.run_evaluation is not None
    assert eval_mod.CitationCoverage is not None
