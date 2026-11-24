PY ?= python

.PHONY: install test example numbers check table1 table3 single-state appendix reproduce clean

install:          ## editable install with test extras
	$(PY) -m pip install -e ".[test]"

test:             ## theory checks on SiouxFalls and the number generator
	$(PY) -m pytest -q

example:          ## silent failure and EPDA on one equilibrium state
	$(PY) examples/quickstart.py

numbers:          ## regenerate the paper numbers from results/ and diff against paper/expected/
	$(PY) paper/make_numbers.py results paper/generated > /dev/null
	diff -u paper/expected/numbers.tex paper/generated/numbers.tex
	diff -u paper/expected/appendix_results.tex paper/generated/appendix_results.tex
	@echo "numbers.tex and appendix_results.tex match the paper"

check:            ## rerun Table 2 and one Table 3 cell and compare with the archived records
	bash experiments/jobs/smoke.sh

table1:           ## rerun Table 1 into outputs/table1
	bash experiments/jobs/table1.sh
table3:           ## rerun Table 3 into outputs/table3
	bash experiments/jobs/table3.sh
single-state:     ## rerun the single-state probes into outputs/single_state
	bash experiments/jobs/single_state.sh
appendix:         ## rerun the Appendix D experiments into outputs/appendix
	bash experiments/jobs/appendix.sh
reproduce: table1 table3 single-state appendix   ## everything (several hours)

clean:
	rm -rf paper/generated outputs .pytest_cache
	find . -name __pycache__ -prune -exec rm -rf {} +
