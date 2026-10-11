import pandas as pd
import pytest

from bmpb.data.corrections import NEW_URL, OLD_URL, TITLE, apply_corrections
from bmpb.data.groups import source_groups


def test_verified_url_repair_preserves_labels_and_original_url():
    frame = pd.DataFrame([dict(item_id="Image_29", title=TITLE, source_url=OLD_URL, label=2)])
    repaired = apply_corrections(frame)
    assert repaired.loc[0, "source_url"] == NEW_URL
    assert repaired.loc[0, "source_url_original"] == OLD_URL
    assert repaired.loc[0, "label"] == 2
    assert frame.loc[0, "source_url"] == OLD_URL
    pd.testing.assert_frame_equal(apply_corrections(repaired), repaired)
    frame.loc[0, "title"] = "different article"
    with pytest.raises(ValueError):
        apply_corrections(frame)


def test_transitive_source_groups_and_missing_urls():
    frame = pd.DataFrame(
        dict(
            item_id=list("abcde"),
            source_index=[1, 1, 2, None, None],
            source_url=["https://example.org/a", None, "https://example.org/a#photo", None, None],
        )
    )
    groups = source_groups(frame)
    assert groups.iloc[0] == groups.iloc[1] == groups.iloc[2]
    assert groups.iloc[3] != groups.iloc[4]
    shuffled = frame.sample(frac=1, random_state=3)
    pd.testing.assert_series_equal(source_groups(shuffled).sort_index(), groups)
