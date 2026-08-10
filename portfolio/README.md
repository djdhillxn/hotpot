# HotpotQA portfolio export

The evaluator writes its complete analysis artifact to
`portfolio/portfolio_trajectories.json`. That file is intentionally exhaustive:
it can include raw model turns, rank lists, active-evidence snapshots, retrieval
telemetry, and every rendered observation.

The GitHub Pages quiz does not need those internals. Build its compact browser
artifact with:

```bash
python3 portfolio/export_quiz_data.py \
  portfolio/portfolio_trajectories.json \
  ../djdhillxn.github.io/assets/json/hotpot/quiz.json
```

By default the exporter retains every question but caps each displayed
observation at 900 characters, preserving both the beginning and end. Use
`--observation-chars` to change that boundary, `--limit` for a smaller curated
artifact, and `--pretty` when the output is intended for manual inspection.

The compact schema contains run-level metrics and an `examples` array. Each
example keeps only what the quiz renders:

- question type and difficulty;
- gold and agent answers;
- answer, supporting-fact, and joint EM/F1;
- visited pages and predicted supporting facts; and
- Thought, Action, and Observation steps.

The page implementation lives in the sibling portfolio repository:

```text
_projects/hotpot.md
assets/css/hotpot/project.css
assets/js/hotpot/quiz.js
assets/json/hotpot/quiz.json
```

Until the full validation run is exported, `quiz.json` is explicitly marked as
demo data and the page reports every aggregate metric as pending.
