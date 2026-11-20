"""A fusion-encoder classifier must depend on the image.

Regression test for the BLIP baseline: AutoModel loaded BlipModel (a dual
encoder) and the classifier read its image-blind text [CLS], so real, blank
and shuffled photos gave identical predictions. Uses a tiny randomly
initialised BLIP saved to a temp dir (synthetic weights, no download).
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from bmpb.config import ExperimentConfig  # noqa: E402
from bmpb.models import multimodal  # noqa: E402


def _tiny_blip(path):
    from transformers import BlipConfig, BlipForImageTextRetrieval

    config = BlipConfig(
        text_config=dict(vocab_size=100, hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=37,
                         encoder_hidden_size=32),
        vision_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                           intermediate_size=37, image_size=32, patch_size=8),
        projection_dim=16,
    )
    torch.manual_seed(0)
    model = BlipForImageTextRetrieval(config)
    # Default init leaves the tiny model's cross-attention path near-inert;
    # random weights everywhere make any image dependence visible.
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.2)
    model.save_pretrained(path)


def test_blip_logits_depend_on_image(tmp_path, monkeypatch):
    _tiny_blip(tmp_path)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        staticmethod(lambda *a, **k: None))
    cfg = ExperimentConfig(name="blip_test", modality="multimodal",
                           family="fusion_encoder", pretrained=str(tmp_path),
                           freeze_encoder=True)
    model, _ = multimodal.fusion_encoder(cfg)
    model.eval()

    input_ids = torch.tensor([[1, 5, 7, 9, 2]])
    attention_mask = torch.ones_like(input_ids)
    torch.manual_seed(1)
    image_a = torch.randn(1, 3, 32, 32)
    image_b = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        logits_a = model(input_ids=input_ids, attention_mask=attention_mask,
                         pixel_values=image_a)["logits"]
        logits_b = model(input_ids=input_ids, attention_mask=attention_mask,
                         pixel_values=image_b)["logits"]
    assert not torch.allclose(logits_a, logits_b), "classifier ignores the image"
