JULIA=julia --startup-file=no

help:
	@echo 'Targets:'
	@echo '    help        - Show this help.'
	@echo '    docs        - Generate docs for webserver deployment.'
	@echo '    localdocs   - Generate docs for local viewing.'
	@echo ''
	@echo 'Variables:'
	@echo '    JULIA       - Controls which command is used to run julia'
	@echo ''
	@echo 'Variables can be set on the commandline using the -e flag for make, e.g.'
	@echo '    make localdocs -e JULIA=path/to/julia'

docs:
	$(JULIA) --project=docs docs/make.jl --prettyurls

localdocs:
	$(JULIA) --project=docs docs/make.jl

clean:
	rm -r docs/build

.PHONY: help docs docs-local clean
