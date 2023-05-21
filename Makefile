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
	@echo '    help           - Show this help.'
	@echo '    docs           - Generate docs for webserver deployment.'
	@echo '    localdocs      - Generate docs for local viewing.'
	@echo '    opendocs       - Open documentation in the browser remaking it if necessary.'
	@echo '    benchmark      - Run benchmark suite.'
	@echo '    benchmarkrev   - Run benchmark suite for revision REV.'
	@echo '    benchmarkcmp   - Run benchmark suite comparing TARGET to BASELINE.'
	@echo ''
	@echo 'Variables:'
	@echo '    JULIA       - Controls which command is used to run julia'
	@echo '                  Default $(JULIA_DEFAULT)'
	@echo '    BROWSER     - Sets the command for how to open html files'
	@echo '                  Default: xdg-open if it exists otherwise open'
	@echo '    REV         - Valid Mercurial revision specifier used in benchmarkrev'
	@echo '    TARGET      - Valid Mercurial revision specifier used in benchmarkcmp'
	@echo '                  as the target revision'
	@echo '    BASELINE    - Valid Mercurial revision specifier used in benchmarkcmp'
	@echo '                  as the baseline revision'
	@echo ''
	@echo 'Variables can be set on the commandline using the -e flag for make, e.g.'
	@echo '    make localdocs -e JULIA=path/to/julia'
	@echo 'or as shell environment variables.'

docs: docs/build

localdocs: docs/build-local

opendocs: localdocs
	$(BROWSER) docs/build-local/index.html

cleandocs:
	rm -rf docs/build
	rm -rf docs/build-local	

benchmark:
	$(JULIA) --project=benchmark benchmark/make.jl

benchmarkrev:
	$(JULIA) --project=benchmark benchmark/make.jl --rev $(REV)

benchmarkcmp:
	$(JULIA) --project=benchmark benchmark/make.jl --cmp $(TARGET) $(BASELINE)

cleanbenchmark:
	rm -rf benchmark/results
	rm -f benchmark/tune.json

clean: cleandocs cleanbenchmark

.PHONY: help clean docs localdocs opendocs cleandocs benchmark benchmarkrev benchmarkcmp cleanbenchmark

SRC_DIRS = src docs/src
SRC_FILES_AND_DIRS = $(foreach dir,$(SRC_DIRS),$(shell find $(dir)))
DEP_IGNORE = %/.DS_Store
DOCS_DEPENDENCIES = docs/make.jl $(filter-out $(DEP_IGNORE),$(SRC_FILES_AND_DIRS))
docs/build: $(DOCS_DEPENDENCIES)
	$(JULIA) --project=docs docs/make.jl --build-dir build --prettyurls

docs/build-local: $(DOCS_DEPENDENCIES)
	$(JULIA) --project=docs docs/make.jl --build-dir build-local
