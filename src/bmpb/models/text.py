"""Text-only encoders: BanglaBERT, Bangla ELECTRA, mBERT, XLM-R, mT5."""

from __future__ import annotations

from bmpb.config import ExperimentConfig
from bmpb.models.registry import register


@register("hf_text_classifier")
def hf_text_classifier(cfg: ExperimentConfig):
    """Any encoder with a sequence-classification head.

    Covers BanglaBERT (csebuetnlp/banglabert), Bangla ELECTRA, mBERT and XLM-R —
    they differ only in the checkpoint name, which lives in the YAML.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    revision = cfg.params.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.pretrained, num_labels=cfg.num_labels, revision=revision
    )
    if cfg.freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    return model, tokenizer


@register("hf_seq2seq_classifier")
def hf_seq2seq_classifier(cfg: ExperimentConfig):
    """mT5 framed as classification.

    The published mt5_predictions.csv collapses to a single class ("Neutral" for
    all 47 test rows), which is what a seq2seq model does when it is asked to
    generate a label string from ~200 training examples. Treating mT5 as an
    encoder with a classification head instead of a generator is the fix; set
    `params.as_generator: true` to reproduce the original degenerate setup.
    """
    if cfg.params.get("as_generator"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        revision = cfg.params.get("revision")
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained, revision=revision)
        return AutoModelForSeq2SeqLM.from_pretrained(cfg.pretrained, revision=revision), tokenizer

    from transformers import AutoTokenizer, T5EncoderModel

    from bmpb.models.heads import EncoderClassifier

    revision = cfg.params.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained, revision=revision)
    encoder = T5EncoderModel.from_pretrained(cfg.pretrained, revision=revision)
    model = EncoderClassifier(
        encoder=encoder,
        hidden_size=encoder.config.d_model,
        num_labels=cfg.num_labels,
        pooling="mean",
    )
    return model, tokenizer
