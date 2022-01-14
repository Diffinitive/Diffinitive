JULIA_DEFAULT=julia --startup-file=no
JULIA?=$(JULIA_DEFAULT)

# Set the default browser
WHICH_XDG_OPEN=$(shell which xdg-open)
WHICH_OPEN=$(shell which open)
BROWSER_DEFAULT  = $(if $(WHICH_XDG_OPEN), xdg-open)
BROWSER_DEFAULT := $(if $(BROWSER_DEFAULT), $(BROWSER_DEFAULT), open)
BROWSER?=$(BROWSER_DEFAULT)

help:
	@echo 'Targets:'
	@echo '    help        - Show this help.'
	@echo '    docs        - Generate docs for webserver deployment.'
	@echo '    localdocs   - Generate docs for local viewing.'
	@echo '    opendocs    - Open documentation in the browser.'
	@echo ''
	@echo 'Variables:'
	@echo '    JULIA       - Controls which command is used to run julia'
	@echo '                  Default $(JULIA_DEFAULT)'
	@echo '    BROWSER     - Sets the command for how to open html files'
	@echo '                  Default: xdg-open if it exists otherwise open'
	@echo ''
	@echo 'Variables can be set on the commandline using the -e flag for make, e.g.'
	@echo '    make localdocs -e JULIA=path/to/julia'
	@echo 'or as shell environment variables.'

docs:
	$(JULIA) --project=docs docs/make.jl --prettyurls

localdocs:
	$(JULIA) --project=docs docs/make.jl

opendocs:
	$(BROWSER) docs/build/index.html

clean:
	rm -r docs/build

.PHONY: help docs localdocs opendocs clean

# TODO:
# Make a real target for docs/build
# Possibly store the local and nonlocal in different build folders
