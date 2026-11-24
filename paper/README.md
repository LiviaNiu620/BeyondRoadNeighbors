# Paper numbers and figures

| File | Role |
|---|---|
| `make_numbers.py` | Reads `results/` and writes `numbers.tex` (LaTeX macros for every number in Tables 1 and 3 and the text) and `appendix_results.tex` (Appendix D tables) to `paper/generated/`. |
| `make_figures.py` | Draws Figures 1–5 from `results/` (PDF and PNG). |
| `expected/` | The two generated files as they appear in the paper's LaTeX source. |

`make numbers` regenerates both files and diffs them against `expected/`; `make figures`
redraws the figures into `figures/`. To check numbers from your own reruns, point the script at
a directory with the same layout: `python paper/make_numbers.py outputs-as-results/ /tmp/check`.
