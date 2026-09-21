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

    text_vectorizer = _char_word_vectorizer("count", cfg.params)

    class CountVecViT:
        """sklearn-shaped: fit/predict over (texts, images)."""

        def __init__(self) -> None:
            self.vectorizer = text_vectorizer
            self.classifier = LogisticRegression(
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
            return np.vstack(features) if features else np.empty((0, vision.config.hidden_size))

        def fit(self, texts, images, labels):
            text_features = self.vectorizer.fit_transform(texts).toarray()
            combined = np.hstack([text_features, self.image_features(images)])
            self.classifier.fit(combined, labels)
            return self

        def predict(self, texts, images):
            text_features = self.vectorizer.transform(texts).toarray()
            combined = np.hstack([text_features, self.image_features(images)])
            return self.classifier.predict(combined)

    return CountVecViT(), processor
