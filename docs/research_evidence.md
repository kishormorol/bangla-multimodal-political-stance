# Research evidence and proposed comparisons

This note supports review of the implementation and its claims. It does not
decide the paper's contribution or establish new annotation ground truth.
Sources were searched and checked on 23 October 2026. Reading was targeted
to the relevant methods, results, and release records. No source below is
claimed as a full paper read.

The final review on the same date rechecked the primary passages supporting
the closest comparisons. BanglaBias Section 3 and its annotation appendices
support the 200 article count, event selection, and separate initial annotation
and adjudication roles. Its public
[dataset card](https://huggingface.co/datasets/dipta007/BanglaBias/raw/main/README.md)
lists MIT, which does not settle permissions for added photographs. Liang
Sections 3.1 to 3.4, Appendix B, and the split footnote confirm the five datasets,
explicit targets, image pair construction, and grouping by tweet ID. CANDOR
Sections 3, 6.3.6, 6.5 and 9, including Table 13, describe its news population,
textual controls, and transfer limits. These were passage checks, not full
paper reads or independent replications. The direct source links below
identify the copies consulted.

## Search scope

The search covered Bangla political stance, paired news images and text,
explicit stance targets, and controls for image contribution. Firecrawl's
academic index was used for papers, with Firecrawl page retrieval for primary
publication and release records. The academic queries were:

1. `Bangla Bengali multimodal political stance detection news headline image benchmark dataset`
2. `"Read Between the Lines" "Bangla" "dataset"`
3. `multimodal stance detection benchmark dataset news images text unimodal ablation`
4. `Bangla Bengali political stance news dataset government neutral annotation`
5. `Multimodal Political Bias Identification and Neutralization Bernard Pleimling Kharel Vickery dataset`
6. `Breaking Bias Context\u002dAware Multimodal Framework Detecting Neutralizing Ideological Bias News Mitra`

Query 6 uses JSON escaping for one punctuation character. Decode `\u002d`
when repeating the recorded query.

The first four searches requested seven results each and the last two four
each. They returned 36 records representing 30 unique papers. Related paper
and citation searches were also used. Web queries checked the named papers,
Bangla dataset releases, Motamot, PoliMemeDecode, and VERITE. The search was
a targeted evidence map, not an exhaustive systematic review. Excluding a
dataset here means that its task did not resolve the named gap; it does not
mean the dataset is invalid.

## Evidence nearest to this repository

1. BanglaBias is part of the collection's lineage.

   The published [BanglaBias paper](https://aclanthology.org/2025.banglalp-1.5/)
   concerns 200 article records with the same government stance categories.
   Its methods describe event based selection, two initial annotators, and a
   third adjudicator. The original article context and media affiliations
   enter its annotation account. See the open methods in
   [Section 3](https://arxiv.org/html/2510.03898#S3) and annotation appendices.

   The local provenance report identifies 145 consistent matches after a URL
   repair. BanglaBias can help reconstruct event metadata and inherited
   records, but overlapping items cannot provide an independent test set.
   The local annotation columns do not establish that their positions equal
   the upstream annotator roles. Original and later guidelines must remain
   distinct.

2. Existing multimodal stance datasets challenge a broad novelty claim.

   [Liang and colleagues](https://aclanthology.org/2024.findings-acl.736/)
   published five tweet and image stance datasets in Findings of ACL 2024.
   The relevant methods are Sections 3.1 to 3.3, with target guidelines in
   Appendix B. Their examples have explicit targets, and the collection is
   English social media rather than Bangla news. Multiple images from one
   post can produce multiple examples, which is relevant to dependence.

   The study supports comparison with prior multimodal stance research. Its
   targets and labels do not justify automatic mapping to government stance
   at a Bangladeshi article's publication date. A release or license check
   would be needed before any reuse; none was downloaded for training here.

3. Controls can reveal what a visual branch contributes.

   The [Vasilakes and colleagues manuscript](https://arxiv.org/abs/2501.17654)
   compares tweet and image inputs with noise images and with image inputs
   alone. Its evaluation methodology also separates image text from other
   visual content through OCR and masking. Those passages, rather than only
   its abstract, were checked. The linked copy is an arXiv manuscript; this
   note does not establish a later publication venue.

   The controls provide methodological options, not results about this Bangla
   corpus. A shuffled or masked image condition is justified only after a
   specific uncertainty about image pairing or embedded text is identified.
   Removing visual evidence may also change the information needed for a
   joint story label, so the interpretation must be explicit.

4. Ideology prediction is a different target.

   [Qiu and colleagues](https://aclanthology.org/2022.emnlp-main.659/)
   study multimodal ideology prediction in news and social media. Section 3
   describes source, community, and politician based ideology labels across
   the datasets. Those labels differ from judging an individual story's
   framing toward the incumbent government at publication.

   The paper is relevant background and a warning about source relationships
   in evaluation. Its data should not be merged into the three government
   stance classes without a separately validated mapping.

The published [CANDOR study](https://dl.acm.org/doi/10.1145/3789258) also
examines multimodal ideological bias and neutralization in news. Targeted
reading covered its data construction, image branch controls, and transfer
limitations. Its US news and rewriting task provide competing evidence about
image usefulness in another setting, not an equivalent Bangla government
stance benchmark. The citation already appears in the local bibliography.

## Dataset decisions

BanglaBias belongs in lineage reconstruction and background discussion. Its
overlap excludes treating it as an untouched external evaluation set.
The English stance datasets belong in task comparison and possible method
development under their actual targets. The ideology datasets belong in
background discussion or a separate research question.

The [Bangladesh government sentiment release](https://data.mendeley.com/datasets/gkb624mc88/1)
describes 6,000 Facebook comments about the interim government. Its positive,
negative, and neutral sentiment labels, political period, and absence of
paired images differ from this benchmark. The release description was read;
its original annotation records were not independently audited. It does not
fill the present image contribution gap.

[Motamot](https://arxiv.org/html/2407.19528#S4) uses positive and negative
political sentiment labels. The [PoliMemeDecode study](https://arxiv.org/html/2607.23493#S6)
classifies memes as political or nonpolitical. Both tasks differ from the
current label definition. Their targeted methods and task descriptions were
screened in the audit. Neither is recommended as added training data. No claim
of reusable rights or compatible gold labels follows from that screening.

No dataset addition is presently justified as a substitute for repairing
experiment identity or defining the annotation target. A larger collection
would not itself isolate image contribution or resolve inherited labels.

## Smallest further study

The present comparisons change text representation and image use together.
A separate proposed study would compare the committed character text
representation alone, the same representation with frozen image features,
and image features alone. All conditions must share original item membership,
outer folds, training examples, label IDs, and classifier selection budget.
The text vectorizer must be fitted inside each training fold. Image features
may be cached only through a fixed model and processor revision.

The primary quantity would be the paired change in macro F1 when image
features are added. An interval entirely above zero supports an improvement
on the specified population. Ruling out a useful gain requires an upper
bound below a meaningful effect chosen before running the study. An interval
that includes zero is inconclusive, not evidence that the effect is absent.
The meaningful effect is an unresolved scientific choice, not a number to
select after seeing results.

Existing items have already supported repeated analyses. Repeated seeds do
not create new independent articles or an untouched evaluation set. Any new
generalization claim would require an evaluation population kept out of
development, with related articles, images, and events assigned together.

Independent Bangla review is required before deciding whether existing labels
can be justified from headlines and images. A random pilot of 30 items can
identify frequent problems, with separately sampled difficult cases used for
diagnosis. Zero problems in 30 independent random items would still permit
an error rate of about 9.5 percent at a one sided 95 percent bound. A pilot
cannot certify the collection. Larger annotation or model sample sizes must
follow a stated precision target or a paired effect calculation using pilot
predictions and the actual grouping structure.

The proposed comparison has since been completed locally. Its recorded
configuration, results, independent repeat and proposed manuscript wording
are in [the focused study PR](https://github.com/kishormorol/bangla-multimodal-political-stance/pull/6).
The proposal above records the reasoning before execution. Adopting the new
result remains the author's decision. No additional dataset was combined with
the corpus, and no automated semantic judgment was presented as human review.
