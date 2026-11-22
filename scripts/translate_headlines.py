"""Cache English machine translations of every headline a run can see.

For the translated-text ablation (author's decision, HANDOFF.md, 2026-11-22):
does CLIP's weakness come from its English tokenizer rather than from the
image-text interaction? Translates the corpus headlines (the `text` the matched
sweep uses) and the published augmentation texts, so translated runs contain no
Bangla. Output: data/interim/translations_en.csv (gitignored, like the
headlines themselves), columns bn, en, model.

The translations are machine output, not reference translations; they are a
fixed preprocessing step, decoded deterministically (beam search, no sampling).

    python -u scripts/translate_headlines.py
"""

import argparse

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from bmpb.data.augment import load_published_augmentations
from bmpb.paths import CORPUS, INTERIM

MODEL = "facebook/nllb-200-distilled-600M"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    corpus = pd.read_csv(CORPUS)
    corpus["text"] = corpus["title"]
    augmentations = load_published_augmentations(corpus)
    texts = sorted(set(corpus["text"]) | set(augmentations.get("text", pd.Series(dtype=str))))
    print(f"{len(texts)} unique texts", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang="ben_Beng")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device).eval()
    target = tokenizer.convert_tokens_to_ids("eng_Latn")

    out = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                               max_length=256).to(device)
            generated = model.generate(**inputs, forced_bos_token_id=target, num_beams=4,
                                       do_sample=False, max_new_tokens=128)
            out += tokenizer.batch_decode(generated, skip_special_tokens=True)
            print(f"{start + len(batch)}/{len(texts)}", flush=True)

    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / "translations_en.csv"
    pd.DataFrame({"bn": texts, "en": out, "model": MODEL}).to_csv(path, index=False)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
