# Annotation guidelines — Bangla multimodal political bias

Version 1.0 · for the ARR October 2026 expansion round

Each item is one news story: a **headline**, the **article body**, and the
**photograph the outlet published with it**. You assign one of three labels to
the story as a whole, and separately record what the *image alone* conveys.

## The three labels

The question is not "is this claim true" or "do I agree". It is: **whose side
does this story's framing serve?** Judge the framing an ordinary Bangladeshi
reader would perceive, not the underlying facts.

### 0 — `govt_critique`
The story's framing puts the government, its institutions, or its allies in a
negative light, or amplifies criticism of them. Signals:

- Reports failure, corruption, repression, or broken promises by the state
- Foregrounds opposition or victim voices making accusations
- Word choice that assigns blame to the state (`দমন–পীড়ন`, `ব্যর্থতা`, `লুটপাট`)
- Gives critics the last word, or leaves their charge unanswered

### 1 — `neutral`
The story reports without tilting toward either side. Signals:

- Procedural or factual reporting (schedules, statistics, court dates)
- Both sides given comparable space and comparable scepticism
- Attributed claims without endorsement (`দাবি করেছেন`, `জানিয়েছে`)
- The political actors are incidental to the story

**`neutral` is a real category, not a residual bin.** Use it when the framing
genuinely does not lean — not when you are unsure. If you are unsure, flag the
item instead.

### 2 — `govt_leaning`
The story's framing favours the government or its position. Signals:

- Foregrounds achievement, development, or stability delivered by the state
- Official statements carried without challenge
- Opposition portrayed as disruptive, criminal, or illegitimate
- Word choice that legitimises state action (`সফলতা`, `উন্নয়ন`, `শান্তি ফিরেছে`)

## Judging the image separately

After labelling the story, record what the **photograph on its own** conveys,
ignoring the text. Same three labels, plus `image_uninformative` for a
photograph that carries no political signal (a building, a road, a generic
crowd, a file portrait used neutrally).

Ask: if you saw only this photo, whose side would it seem to be on? A picture of
a politician mid-shout at a rally reads differently from the same politician
seated at a desk.

This second judgement is not redundant. In the current 198-item set, the story
label and the image label differ on **about half** of the items that carry both.
That disagreement is a finding the paper reports, so record what you actually
see rather than copying the story label.

## Decision procedure

1. Read the **headline** first and form an impression.
2. Read the **body**. If it changes your impression, the body wins — it is the
   fuller evidence.
3. Assign the story label.
4. Look at the **image alone** and assign the image label.
5. If the item is not about Bangladeshi politics at all, mark `not_political`
   rather than `neutral`. These are filtered out, not trained on.
6. If you cannot decide between two labels, pick the more likely one **and
   flag** the item with a one-line reason. Flagged items go to adjudication.

Do not look up the outlet's reputation, and do not label from the outlet name.
Outlet identity is metadata we analyse afterwards; if it drives the labels, that
analysis becomes circular.

## Hard cases, and how to resolve them

**An opposition figure is accused of a crime by the state.**
If the framing treats the accusation as established, that favours the
government's narrative → `govt_leaning`. If it foregrounds the accused's denial
or questions the process → `govt_critique`. If it reports the charge and the
denial evenhandedly → `neutral`.

**A protest is reported.**
Focus on whose grievance is centred. Protesters' demands carried
sympathetically → `govt_critique`. Emphasis on disruption, vandalism, or arrests
→ `govt_leaning`. A bare account of what happened → `neutral`.

**Criticism of a *past* government.**
Label relative to the **current** government at the article's publication date.
Criticism of a predecessor that the incumbent also criticises serves the
incumbent's framing → `govt_leaning`.

**An opinion column.**
Label the stance the column argues for. Opinion pieces are legitimate items;
`articleSection` records that they are opinion.

**A story that is critical of everyone.**
If criticism lands on the government and the opposition in comparable measure →
`neutral`.

**A wire or agency report.**
Label what is on the page. Do not reason about the agency's intent.

## Flagging

Flag rather than guess when:

- The political direction depends on context you do not have
- The headline and body point in opposite directions
- The item is borderline `neutral` / leaning and you could argue either way
- The image is missing, broken, or unrelated to the story

Flagged items cost far less than silently wrong labels. Roughly 5–10% flagged is
healthy; under 2% usually means guessing.

## What we do with your labels

Three annotators label every item independently. We report pairwise Cohen's κ
per annotator pair, and the final label is the majority of the three. Items
where all three disagree, or that any annotator flagged, go to adjudication and
are resolved in discussion — with the outcome recorded.

Do not discuss specific items with the other annotators while labelling. The
agreement statistic is only meaningful if the passes are independent.

## Known limitation, stated for the record

These guidelines were written after the first 198 items were labelled, by
reconstructing the criteria those labels imply. The expansion round is the first
to use them prospectively. The paper reports this: agreement on the original
subset and on the new subset are given separately, because only the second was
produced under a written protocol.

## Where you work

The annotation desk is at **https://claude.ai/artifact/HkKKt8y9QKrhtcsJ4SYLYa**
(Kishor has to share it with your account before you can open it).

Your labels save automatically as you go, and the desk reopens at the first item
you have not finished. You cannot see the other annotators' labels — that is
deliberate, and it is what makes the agreement figure meaningful.

Keys: `1` `2` `3` for the story stance, `0` for not political, `q` `w` `e` `r`
for the image, `f` to flag, `←` `→` to move. Setting the stance moves you on once
the image is set, so a straightforward item is two keystrokes.
