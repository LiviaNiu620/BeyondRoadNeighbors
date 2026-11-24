# Paper numbers

| File | Role |
|---|---|
| `make_numbers.py` | Reads `results/` and writes `numbers.tex` (LaTeX macros for every number in Tables 1 and 3 and the text) and `appendix_results.tex` (Appendix D tables) to `paper/generated/`. |
| `expected/` | The two generated files as they appear in the paper's LaTeX source. |

`make numbers` regenerates both files and diffs them against `expected/`. To check numbers from
your own reruns, point the script at a directory with the same layout:
`python paper/make_numbers.py my-results/ /tmp/check`.
