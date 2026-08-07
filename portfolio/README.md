# ReAct Agent Portfolio Trajectory Visualizer

This directory provides a self-contained, interactive HTML/JS/CSS component designed for showcasing ReAct reasoning trajectories directly on your GitHub Pages (github.io) or personal website.

---

## Directory Structure

```
portfolio/
├── portfolio_trajectories.json # Exported ReAct trajectories (Thought, Action, Observation, Graph)
├── index.html                  # Standalone interactive trajectory inspector HTML
├── style.css                   # Glassmorphism dark-mode theme stylesheet
├── app.js                      # Dynamic JSON parser & trajectory accordion renderer
└── README.md                   # Integration and embedding guide
```

---

## How It Works

1. Automatic JSON Export: Running the evaluation benchmark script (python eval/run_eval.py --samples 100 --source official_json) automatically exports all generated trajectories to portfolio/portfolio_trajectories.json.
2. Schema Definition:
   ```json
   [
     {
       "id": "sample_1",
       "question": "Were Scott Derrickson and Ed Wood born in the same state?",
       "ground_truth": "no",
       "predicted_answer": "no",
       "exact_match": true,
       "joint_f1": 1.0,
       "step_count": 3,
       "visited_pages": ["Scott Derrickson", "Ed Wood"],
       "evidence_graph": [
         {"source": "Scott Derrickson", "target": "Ed Wood", "label": "Searched 'Ed Wood'"}
       ],
       "steps": [
         {
           "step": 1,
           "thought": "I need to search for Scott Derrickson to find his birthplace.",
           "action": "search[Scott Derrickson]",
           "observation": "Observation: Loaded [Scott Derrickson]: Scott Derrickson was born in Denver, Colorado."
         }
       ]
     }
   ]
   ```

---

## Embedding in Your GitHub.io Portfolio

### Option 1: Direct Link or Subfolder Hosting
Copy the portfolio/ directory into your github.io repository as a subfolder (e.g. yourusername.github.io/react-hotpot-agent/) and host index.html directly via GitHub Pages.

### Option 2: iFrame Embedding
Embed the visualizer into any existing portfolio page using a clean iFrame:

```html
<iframe 
    src="portfolio/index.html" 
    width="100%" 
    height="750px" 
    style="border: 1px solid #30363d; border-radius: 8px;"
></iframe>
```

---

## Viewing Locally

You can open portfolio/index.html in any browser or launch a simple local HTTP server:

```bash
python3 -m http.server 8080 --directory portfolio/
```
Navigate to http://localhost:8080 in your browser.
