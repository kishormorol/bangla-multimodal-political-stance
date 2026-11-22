"""Non-neural and shallow baselines.

With ~200 annotated articles these are not a formality: a TF-IDF classifier is
often within noise of a fine-tuned transformer at this scale, and a
majority-class baseline sets the floor a three-way result must clear (~52% with
this label distribution). Every reported neural number should be read against
these.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bmpb.config import ExperimentConfig
from bmpb.models.registry import register


def _char_word_vectorizer(kind: str, params: dict):
    """Character n-grams matter for Bangla: they survive the inflectional
    morphology and the spelling variation across outlets."""
    analyzer = params.get("analyzer", "char_wb")
    ngram = tuple(params.get("ngram_range", (2, 5) if analyzer.startswith("char") else (1, 2)))
    common = {
        "analyzer": analyzer,
        "ngram_range": ngram,
        "min_df": params.get("min_df", 2),
        "max_features": params.get("max_features", 50_000),
        "sublinear_tf": True,
    }
    if kind == "count":
        common.pop("sublinear_tf")
        return CountVectorizer(**common)
    return TfidfVectorizer(**common)


@register("sklearn_text")
def sklearn_text(cfg: ExperimentConfig):
    """TF-IDF or bag-of-words + a linear classifier. Returns (pipeline, None)."""
    params = cfg.params
    vectorizer = _char_word_vectorizer(params.get("vectorizer", "tfidf"), params)
    classifier_name = params.get("classifier", "logreg")
    if classifier_name == "svm":
        classifier = LinearSVC(
            C=params.get("C", 1.0), class_weight="balanced" if cfg.class_weights else None
        )
    else:
        classifier = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 2000),
            class_weight="balanced" if cfg.class_weights else None,
        )
    return Pipeline([("features", vectorizer), ("classifier", classifier)]), None


@register("majority")
def majority(cfg: ExperimentConfig):
    """The floor. `strategy: stratified` in params gives the random-guess floor."""
    strategy = cfg.params.get("strategy", "most_frequent")
    return DummyClassifier(strategy=strategy, random_state=cfg.seed), None


@register("countvec_vit")
def countvec_vit(cfg: ExperimentConfig):
    """Bag-of-words text features concatenated with frozen ViT image features.

    This is the shallow multimodal baseline from the original experiments: it
    isolates how much the image contributes when the two modalities never
    actually interact.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    vision_name = cfg.params.get("vision_model", "google/vit-base-patch16-224-in21k")
    processor = AutoImageProcessor.from_pretrained(vision_name)
    vision = AutoModel.from_pretrained(vision_name)
    vision.eval()
    for param in vision.parameters():
        param.requires_grad = False

    # text_features: "count" (the original baseline), "tfidf" (the same
    # char n-gram TF-IDF as tfidf_logreg, so the pair differs only by the
    # image), or "none" (image-only).
    text_kind = cfg.params.get("text_features", "count")
    if text_kind not in {"count", "tfidf", "none"}:
        raise ValueError(f"text_features must be count, tfidf or none, not {text_kind!r}")
    text_vectorizer = None if text_kind == "none" else _char_word_vectorizer(text_kind, cfg.params)
    # image_norm: "none" (raw ViT [CLS], the original baseline) or "l2" (each
    # image vector scaled to unit length, matching the L2-normalised TF-IDF
    # rows, so neither block dominates the logistic regression by scale alone).
    image_norm = cfg.params.get("image_norm", "none")
    if image_norm not in {"none", "l2"}:
        raise ValueError(f"image_norm must be none or l2, not {image_norm!r}")

    class CountVecViT:
        """sklearn-shaped: fit/predict over (texts, images)."""

        def __init__(self) -> None:
            self.vectorizer = text_vectorizer
            self.classifier = LogisticRegression(
                C=cfg.params.get("C", 1.0),
                max_iter=cfg.params.get("max_iter", 2000),
                class_weight="balanced" if cfg.class_weights else None,
            )
            self.processor = processor
            self.vision = vision

        def image_features(self, images) -> np.ndarray:
            features = []
            batch = cfg.eval_batch_size
            with torch.no_grad():
                for start in range(0, len(images), batch):
                    chunk = images[start : start + batch]
                    inputs = self.processor(images=chunk, return_tensors="pt")
                    pooled = self.vision(**inputs).last_hidden_state[:, 0]
                    features.append(pooled.cpu().numpy())
            if not features:
                return np.empty((0, vision.config.hidden_size))
            stacked = np.vstack(features)
            if image_norm == "l2":
                norms = np.linalg.norm(stacked, axis=1, keepdims=True)
                stacked = stacked / np.maximum(norms, 1e-12)
            return stacked

        def features(self, texts, images, fit=False):
            image = self.image_features(images)
            if self.vectorizer is None:
                return image
            vectorize = self.vectorizer.fit_transform if fit else self.vectorizer.transform
            return np.hstack([vectorize(texts).toarray(), image])

        def fit(self, texts, images, labels):
            self.classifier.fit(self.features(texts, images, fit=True), labels)
            return self

        def predict(self, texts, images):
            return self.classifier.predict(self.features(texts, images))

    return CountVecViT(), processor
