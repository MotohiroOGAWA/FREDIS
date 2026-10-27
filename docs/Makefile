SPHINXBUILD ?= python -m sphinx
SOURCEDIR = .
BUILDDIR = _build

.PHONY: html check clean
html:
	$(SPHINXBUILD) -W --keep-going -b html $(SOURCEDIR) $(BUILDDIR)/html
check:
	$(SPHINXBUILD) -W --keep-going -b dummy $(SOURCEDIR) $(BUILDDIR)/check
clean:
	$(SPHINXBUILD) -M clean $(SOURCEDIR) $(BUILDDIR)
