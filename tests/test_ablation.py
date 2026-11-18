"""Image ablation: blank and shuffled photos, and the new configs."""

import numpy as np
import pandas as pd
import pytest

from bmpb.config import ExperimentConfig
from bmpb.crossval import ablate_images
from bmpb.data.dataset import BLANK_IMAGE, load_image
from bmpb.paths import CONFIGS


def corpus(n=10):
    return pd.DataFrame({"item_id": [f"i{k}" for k in range(n)],
                         "image_path": [f"images/{k}.jpg" for k in range(n)]})


def test_blank_gives_every_item_the_same_grey_image():
    out = ablate_images(corpus(), "blank", seed=1)
    assert (out.image_path == BLANK_IMAGE).all()
    image = load_image(BLANK_IMAGE, size=32)
    assert image.size == (32, 32)
    assert set(np.asarray(image).ravel()) == {128}


@pytest.mark.parametrize("seed", range(20))
def test_shuffled_is_a_derangement_of_the_same_photos(seed):
    original = corpus()
    out = ablate_images(original, "shuffled", seed=seed)
    assert sorted(out.image_path) == sorted(original.image_path)
    assert (out.image_path.to_numpy() != original.image_path.to_numpy()).all()
    assert list(out.item_id) == list(original.item_id)  # folds are unaffected


def test_shuffle_is_seeded():
    a = ablate_images(corpus(), "shuffled", seed=3).image_path.tolist()
    b = ablate_images(corpus(), "shuffled", seed=3).image_path.tolist()
    assert a == b


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        ablate_images(corpus(), "noise", seed=0)


def test_ablation_configs_differ_from_their_parent_only_in_name_and_ablation():
    for parent in ["countvec_vit", "clip", "align", "blip", "vilt", "flava"]:
        base = vars(ExperimentConfig.load(CONFIGS / "multimodal" / f"{parent}.yaml"))
        for mode in ["blank", "shuffled"]:
            child = vars(ExperimentConfig.load(CONFIGS / "multimodal" / f"{parent}_{mode}.yaml"))
            assert child.pop("name") == f"{parent}_{mode}"
            assert child.pop("image_ablation") == mode
            assert child == {k: v for k, v in base.items() if k not in {"name", "image_ablation"}}


def test_tfidf_vit_uses_exactly_the_tfidf_logreg_text_features():
    text = ExperimentConfig.load(CONFIGS / "text" / "tfidf_logreg.yaml").params
    joint = ExperimentConfig.load(CONFIGS / "multimodal" / "tfidf_vit.yaml").params
    assert joint["text_features"] == "tfidf"
    for key in ["analyzer", "ngram_range", "min_df", "max_features", "C"]:
        assert joint[key] == text[key], key
    shuffled = ExperimentConfig.load(CONFIGS / "multimodal" / "tfidf_vit_shuffled.yaml")
    assert shuffled.image_ablation == "shuffled" and shuffled.params == joint
