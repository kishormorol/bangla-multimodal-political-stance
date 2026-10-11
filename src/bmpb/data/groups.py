"""Conservative source components shared by holdout and cross-validation."""

import unicodedata
from urllib.parse import unquote, urlsplit, urlunsplit

import pandas as pd


def normalized_url(value):
    if pd.isna(value) or not str(value).strip():
        return None
    value = unicodedata.normalize("NFC", unquote(str(value).strip()))
    parts = urlsplit(value)
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, "")
    )


def source_groups(frame, group_key="source_index"):
    """Join rows sharing a declared group, source URL, or augmentation parent.

    Missing metadata never joins unrelated rows. Components are transitive and
    named by item ID, making their names independent of row order.
    """
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(a, b):
        parent[find(b)] = find(a)

    seen = {}
    for i, (_, row) in enumerate(frame.iterrows()):
        keys = [("item", str(row["item_id"]))]
        for column in dict.fromkeys((group_key, "source_group", "source_index")):
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                keys.append((column, str(value)))
        url = normalized_url(row.get("source_url"))
        if url:
            keys.append(("url", url))
        value = row.get("parent_item_id")
        if pd.notna(value) and str(value).strip():
            keys.append(("item", str(value)))
        for key in keys:
            if key in seen:
                join(i, seen[key])
            else:
                seen[key] = i
    names = {}
    for i, item in enumerate(frame.item_id.astype(str)):
        component = find(i)
        names[component] = min(names.get(component, item), item)
    return pd.Series(["source:" + names[find(i)] for i in range(len(frame))], index=frame.index)
