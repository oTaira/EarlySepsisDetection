# CS230 Autumn 2026 milestone draft

`main.tex` is a review draft for Nick Allen and Tyler Yamori-Little. It compiles in Codex's built-in LaTeX editor.

This folder uses the provided `milestone_template.zip`: its official NeurIPS 2026 style and bibliography setup, with the CS230 milestone sections filled in. The report must fit within three pages excluding references. Compile `main.tex` with BibTeX and verify the page count before submission.

## Review before submission

- Confirm both authors' full names and SUNet IDs. The CS230 proposal file calls Nick `Nicholas Simon`; the CS224W proposal calls him `Nick Allen`. The draft uses `Nick Allen` pending confirmation.
- Have Tyler verify every method description and rounded metric against the committed run files.
- The draft reports only past-only models. The ensemble is the causal probability blend in `runs/ensemble_causal/metrics.json` (test utility 0.418). Do not put the earlier linear-interpolation scores back into the table.
- The test set has been examined across many runs. Do not use it for further model selection or claim comparison with the competition's different hidden cohort.
- Verify the CS230/CS224W shared-project approval and attribution before submission. The milestone text identifies the separate scopes.
- Submit one group PDF to Gradescope, tag teammates and pages, by November 10, 2026 at 11:59 PM PST.
