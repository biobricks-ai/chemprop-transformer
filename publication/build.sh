#!/bin/bash

# Compile the main LaTeX document
pdflatex main.tex

# Run BibTeX if you have a bibliography
bibtex main

# Compile twice more to resolve references
pdflatex main.tex
pdflatex main.tex

# Clean up auxiliary files
rm -f *.aux *.log *.out *.toc *.bbl *.blg