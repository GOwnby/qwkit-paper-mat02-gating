#!/bin/sh
# Regenerate every MAT-02 number and figure (paper_mat02/). About 6 h on one core (uq_xmax.py about 4 h).
set -e
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd) OMP_NUM_THREADS=1
cd "$(dirname "$0")/mat02"
python3 -W ignore gate_map.py    > gate_map.log      # x_max, surface-donor share, double-layer offset (Table 1)
python3 -W ignore gate_curves.py > gate_curves.log   # x_max(n_s), offset(n_s), field-free margin (Fig. 2)
python3 -W ignore donor_layer.py > donor_layer.log   # uniformly doped layer next to the interface
python3 -W ignore profiles.py    > profiles.log      # band profiles (Fig. 1)
python3 -W ignore spectra.py     > spectra.log       # ISB observables per placement (Fig. 3)
python3 -W ignore uq_xmax.py     > uq_xmax.log       # parameter uncertainty (Fig. 4; needs studies/mat01/uq.py)
python3 collect_numbers.py > numbers.log
python3 tables.py > tables.log
python3 paper_figures.py
cd ../../paper_mat02 && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && \
  pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
cp main.pdf ../MAT02_paper.pdf
