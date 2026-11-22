"""Follow-up ablations: L2-normalised image features and translated text."""

import pandas as pd
import pytest
import yaml

from bmpb.crossval import translate_frames
from bmpb.paths import CONFIGS


def _load(name):
    return yaml.safe_load((CONFIGS / "multimodal" / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("child", "parent", "changed"),
    [
        ("tfidf_vit_l2", "tfidf_vit", {"params.image_norm"}),
        ("tfidf_vit_l2_shuffled", "tfidf_vit_l2", {"image_ablation"}),
        ("clip_en", "clip", {"params.translate_text"}),
        ("clip_en_blank", "clip_en", {"image_ablation"}),
    ],
)
def test_followup_configs_change_one_thing(child, parent, changed):
    def flat(cfg):
        out = {f"params.{k}": v for k, v in cfg.pop("params", {}).items()}
        out.update(cfg)
        out.pop("name")
        return out

    a, b = flat(_load(child)), flat(_load(parent))
    diff = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert diff == changed


def test_translate_frames_maps_every_text(tmp_path):
    cache = tmp_path / "t.csv"
    pd.DataFrame({"bn": ["ক", "খ"], "en": ["a", "b"]}).to_csv(cache, index=False)
    corpus = pd.DataFrame({"item_id": [1, 2], "text": ["ক", "খ"]})
    aug = pd.DataFrame({"item_id": ["1__aug0"], "text": ["খ"]})
    c, a = translate_frames(corpus, aug, cache)
    assert c["text"].tolist() == ["a", "b"] and a["text"].tolist() == ["b"]
    assert corpus["text"].tolist() == ["ক", "খ"]  # inputs untouched


def test_translate_frames_refuses_missing_translation(tmp_path):
    cache = tmp_path / "t.csv"
    pd.DataFrame({"bn": ["ক"], "en": ["a"]}).to_csv(cache, index=False)
    corpus = pd.DataFrame({"item_id": [1, 2], "text": ["ক", "গ"]})
    with pytest.raises(KeyError):
        translate_frames(corpus, pd.DataFrame(), cache)
