import numpy as np
import pandas as pd
import pytest

from bmpb.config import ExperimentConfig
from bmpb.crossval import _fit_predict, cross_validate, make_folds


def test_neural_cv_requires_independent_validation(monkeypatch):
    from bmpb import train

    seen = {}

    def fake_train(cfg, splits, out):
        seen.update(splits)
        return splits["test"], np.array([0]), None

    monkeypatch.setattr(train, "_train_torch", fake_train)
    cfg = ExperimentConfig(name="test", modality="text", family="hf_text_classifier")
    training = pd.DataFrame({"item_id": ["a"], "label": [0]})
    validation = pd.DataFrame({"item_id": ["b"], "label": [0]})
    test = pd.DataFrame({"item_id": ["c"], "label": [0]})
    _fit_predict(cfg, training, test, validation)
    assert set(seen["val"].item_id).isdisjoint(seen["test"].item_id)


def test_outer_test_never_selects_checkpoints(tmp_path, monkeypatch):
    from bmpb import crossval

    corpus = pd.DataFrame(
        {
            "item_id": [f"item{i}" for i in range(60)],
            "source_index": np.repeat(np.arange(30), 2),
            "label": np.repeat(np.arange(30) % 3, 2),
        }
    )
    config = tmp_path / "config.yaml"
    config.write_text("name: test\nmodality: text\nfamily: hf_text_classifier\n")
    monkeypatch.setattr(crossval, "load_published_augmentations", lambda df: pd.DataFrame())

    def predict(cfg, train, test, val):
        assert val is not None and not val.empty
        groups = [set(frame.source_index) for frame in (train, val, test)]
        assert not groups[0] & groups[1]
        assert not groups[0] & groups[2]
        assert not groups[1] & groups[2]
        return test, test.label.to_numpy(), None

    monkeypatch.setattr(crossval, "_fit_predict", predict)
    summary = cross_validate(config, corpus=corpus, out=tmp_path)
    assert summary["items_scored"] == len(corpus)
    assert all(fold["n_val"] > 0 for fold in summary["folds"])
    assert "git_commit" in summary


def test_fold_count_must_allow_held_out_groups():
    corpus = pd.DataFrame({"item_id": ["a", "b"], "label": [0, 1]})
    for count in (1, 3):
        with pytest.raises(ValueError):
            make_folds(corpus, n_splits=count)


def test_duplicate_urls_stay_together_without_source_index():
    corpus = pd.DataFrame({
        "item_id": [f"item{i}" for i in range(30)],
        "label": np.arange(30) % 3,
        "source_url": [f"https://news.example/story/{i}" for i in range(30)],
    })
    corpus.loc[1, "source_url"] = corpus.loc[0, "source_url"] + "#photo"
    corpus.loc[1, "label"] = corpus.loc[0, "label"]
    folds = make_folds(corpus)
    memberships = {int(row): fold for fold, rows in enumerate(folds) for row in rows}
    assert memberships[0] == memberships[1]
