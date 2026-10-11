# Dataset provenance audit

Completed October 10, 2026 using the local raw and processed tables and the
public [BanglaBias dataset](https://huggingface.co/datasets/dipta007/BanglaBias),
which accompanies [Lia et al. (2025)](https://aclanthology.org/2025.banglalp-1.5/).

## Finding

**This collection contains reused BanglaBias items. It is not an independently
collected 198-item dataset.** The raw sheet explicitly includes a column named
`id from bangla bias`. Row-level comparison establishes overlap, not merely
similar task definitions or sample sizes.

| Evidence | Raw sheet (200 items) | Processed corpus (198 items) |
| --- | ---: | ---: |
| Headline and URL agree with the same upstream candidate | 135 | 134 |
| URL match only | 5 | 5 |
| Headline match only | 6 | 6 |
| Headline and URL identify different upstream candidates | 1 | 0 |
| Neither headline nor URL matches | 53 | 53 |

After the verified Image_29 URL repair, 145 processed items have consistent matching evidence. The raw sheet retains its original metadata conflict. The 53 unmatched items should be
described as **not matched to this upstream snapshot**, rather than automatically
claimed as independently collected or newly annotated. Establishing their origin
still needs collection records. No fuzzy matching was used.

The matcher normalizes Unicode to NFC and collapses whitespace in headlines.
For URLs it additionally decodes percent escapes and removes a trailing slash.
It preserves all candidate IDs when the upstream table contains duplicate
headlines or URLs, rather than silently selecting the last occurrence. The
matching fields were not truncated in the dataset-server response.

## Exceptions requiring data repair or author records

- **Image_29:** headline identifies upstream ID 160, but source URL identifies
  upstream ID 157. Its declared upstream ID is 160. The source URL has been repaired using the verified Prothom Alo headline. Labels were preserved; old body caches are not attached to repaired URLs.
- **Image_44 and Image_115:** more than one upstream ID remains possible because
  the upstream resource duplicates headlines or URLs. Preserve candidate IDs;
  do not treat an arbitrary match as a verified one-to-one lineage.
- **Image_29 / Image_32:** same local source URL, different headlines and final
  labels. The verified Image_29 repair removes this URL collision.
- **Image_154 / Image_155:** same local source URL, different headlines, same
  final critique label. Establish whether these are distinct image pairings of
  one story or a source-link error. Until resolved, keep them in one source group.
- **Image_118 and Image_152:** present in the raw sheet but absent from the
  processed corpus. Both match upstream content. Their exclusion rationale is
  not established by the row comparison; the old manuscript's deduplication
  explanation needs the original processing records.

The shared split and CV logic now joins declared groups, normalized source URLs
(including fragment variants), and augmentation parents transitively. Regression
tests reproduce and prevent the original leakage. Image_29's source URL has been
corrected to the verified [Prothom Alo article](https://www.prothomalo.com/business/economics/7p5t31msj0),
whose headline matches upstream ID 160. Its labels and image mapping are unchanged;
`source_url_original` retains the old URL. Image_154/Image_155 remain in one group.
The baseline reruns are recorded in `reports/submission/baselines.md`; historical
neural scores still require rerunning.

## Supported manuscript wording

> We assemble a multimodal working collection that includes items from
> BanglaBias (Lia et al., 2025). Our processed table contains 198 news items.
> A row-level provenance audit identifies 145 items with consistent matches
> to the public BanglaBias snapshot after one verified URL correction,
> and 53 items without an exact normalized match.
> The table includes paired-image metadata and separate image stance fields.
> We distinguish the original article annotations from the final item labels.

This wording establishes reuse and describes fields actually present. It does
not claim that all images or annotations were newly collected. Before a final
dataset-contribution claim, document who obtained the images, how image and
final-item labels were produced, the additional items' origin, and permissions.

The original resource is article-level. Training on headlines while retaining
article-derived labels evaluates prediction of those labels from headlines; it
does not establish independently annotated headline stance. Specify what the
annotators saw before calling the labels headline-level ground truth.

## Attribution and licensing

Cite the published BLP paper, rather than describing this as the first Bangla
political stance dataset. Its Hugging Face card lists MIT, but the local added
image assets and annotations need their own documented terms. The upstream
card does not establish permissions for third-party photographs or for
redistribution of added news content.

## Reproduction

`reports/submission/provenance.json` stores match counts, candidate upstream
IDs, exception item IDs, and hashes of all three input tables. It contains no
article bodies or photographs. The local exported upstream table and card are
stored under ignored `data/raw/`.

```bash
.venv/bin/python scripts/audit_provenance.py \
  --upstream data/raw/banglabias-upstream.json
```

The source export came from the dataset-server `default` / `test` split in
two 100-row pages. The two pages together contain all 200 upstream rows.
