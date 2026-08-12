# HotpotQA portfolio export

The evaluator writes its complete ReAct execution artifact to
`eval_results/react/trajectories.json`. That file is intentionally exhaustive:
it can include raw model turns, rank lists, active-evidence snapshots, retrieval
telemetry, and every rendered observation.

The GitHub Pages quiz does not need those internals. Build its compact browser
artifact with:

```bash
python3 portfolio/export_quiz_data.py \
  eval_results/react/trajectories.json \
  ../djdhillxn.github.io/assets/json/hotpot/react_trajectories.json
```

By default the exporter retains every question and every displayed observation
in full. Use `--observation-chars` only when an explicitly shortened build is
needed, `--limit` for a smaller development fixture, and `--pretty` when the
output is intended for manual inspection.

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
assets/json/hotpot/react_trajectories.json
```
