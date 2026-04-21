PAPER = infogate

.PHONY: all clean watch view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex refs.bib $(wildcard sections/*.tex) $(wildcard tables/*.tex) $(wildcard figures/*.tex) $(wildcard figures/*.pdf) $(wildcard figures/*.png)
	pdflatex -interaction=nonstopmode $(PAPER).tex >/dev/null || true
	bibtex $(PAPER) || true
	pdflatex -interaction=nonstopmode $(PAPER).tex >/dev/null || true
	pdflatex -interaction=nonstopmode $(PAPER).tex >/dev/null || true
	@if [ -f $(PAPER).pdf ]; then echo "==== build OK: $(PAPER).pdf ===="; else echo "==== build FAILED: $(PAPER).pdf missing ===="; tail -40 $(PAPER).log; exit 1; fi
	@grep -E "^!|LaTeX Error|Undefined" $(PAPER).log | head -10 || true
	@grep -E "Warning|Underfull|Overfull" $(PAPER).log | head -20 || true

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).toc $(PAPER).fdb_latexmk $(PAPER).fls $(PAPER).synctex.gz
	rm -f sections/*.aux figures/*.aux

view: $(PAPER).pdf
	@echo "open $(PWD)/$(PAPER).pdf in your viewer"
