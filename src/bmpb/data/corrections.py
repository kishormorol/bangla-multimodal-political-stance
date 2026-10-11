"""Audited metadata repairs; preserve original source sheets and all labels."""

ITEM = "Image_29"
TITLE = "রামপালে হচ্ছে বড় সৌরবিদ্যুৎকেন্দ্র"
OLD_URL = "https://www.jugantor.com/budget/681279"
NEW_URL = "https://www.prothomalo.com/business/economics/7p5t31msj0"


def apply_corrections(frame):
    frame = frame.copy()
    selected = frame.item_id.eq(ITEM)
    if not selected.any():
        return frame
    row = frame.loc[selected].iloc[0]
    if row.title != TITLE or row.source_url not in (OLD_URL, NEW_URL):
        raise ValueError("Image_29 metadata differs from the audited correction")
    frame.loc[selected, "source_url_original"] = OLD_URL
    frame.loc[selected, "source_url"] = NEW_URL
    frame.loc[selected, "source_correction_evidence"] = (
        "BanglaBias ID 160; verified Prothom Alo headline"
    )
    return frame
