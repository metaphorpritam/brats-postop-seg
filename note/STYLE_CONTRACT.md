# Style & notation contract — BraTS explainer note

Every section obeys this so the assembled document reads as one voice and renders cleanly.

## Audience & depth
- Reader is an **intelligent layperson**: no medical training, no deep-learning background.
- **Define every term at first use.** Build from basics. Never write "it can be shown that" —
  show it. Derivations are **step by step with NO skipped algebra**: each step is one equation
  plus a plain-English clause saying what changed and why. Explain every variable.
- Aim ~700–1400 words per section: prose + 1–3 key equations (with a full derivation where the
  topic has one) + at least one **worked numeric example** (substitute real small numbers) +
  1 Intuition + 1 Gotcha + 1–2 Example-question boxes.

## Heading
- Start the file with exactly one `## N. Title {#id}` line (the number, title, and id are given
  to you — use them verbatim). Use `###` for subsections, `####` sparingly. No `#` (that is the
  document title) and no other `##`.

## Math (MathJax via pymdownx.arithmatex)
- Inline math: `$ ... $`  ·  Display math: `$$ ... $$` on its own lines.
- Use standard LaTeX. Multi-line derivations: use `$$\begin{aligned} ... \\ ... \end{aligned}$$`.
- Define each symbol the first time it appears, in prose right after the equation.
- Do NOT put a stray `$` in ordinary prose (write "5 dollars", not "\$5") — it starts math mode.

## Callout boxes (admonitions) — 4-space indented body, blank line before
```
!!! intuition "Intuition"
    Plain-English picture behind the formula/choice.

!!! gotcha "Gotcha"
    A trap: where the obvious reading is wrong, or an assumption quietly fails.

!!! example "Example question"
    A short worked problem. **Solution:** substitute numbers step by step to the answer.
```
Use these three types only. Give ≥1 gotcha, ≥1 intuition, ≥1–2 example questions per section.

## Diagrams
- Do NOT put ```mermaid fences in the markdown. Instead, RETURN diagram specs separately
  (name + mermaid code + alt/caption). In the markdown, reference the rendered PNG as:
  `![caption text](figures/diagrams/NAME.png)` where NAME matches the diagram you return.
- Keep diagrams simple so they never overlap: short node labels, `<br/>` for line breaks,
  `flowchart TB` or `LR`, ≤ ~8 nodes. Prefer one clear diagram over a busy one.

## Tables
- Use markdown tables for comparisons (e.g., modality vs. what-it-shows). Keep cells short.

## Citations (critical for medical claims)
- The research JSON gives `key_points` each with `citation_url` + `source`. For every medical/
  clinical claim you state, add an inline link to its source: `... fact [NCI](https://...).`
- Never state a medical fact that is not backed by a citation in the research JSON. If the JSON
  flags something as unverified (`confidence_notes`), either omit it or hedge explicitly.

## Cross-references
- Refer to other sections as `§N` using this map (do not invent section numbers):
  §2 brain tumours · §3 gliomas/WHO grading · §4 glioblastoma & treatment · §5 the post-treatment
  brain · §6 MRI modalities · §7 the 5 BraTS labels · §8 clinical impact · §9 segmentation as a task
  · §10 CNNs/convolutions · §11 the U-Net · §12 softmax & cross-entropy · §13 Dice & Dice loss ·
  §14 class imbalance · §15 normalization · §16 evaluation metrics · §17 training machinery ·
  §18 controlled A/B design · §19 this project's dataset & design · §20 the defects D1–D9 ·
  §21 results & interpretation · §22 takeaways & limitations.

## Voice
- Precise, warm, textbook-plus. Short paragraphs. Bold the key term when introduced. No hype.
