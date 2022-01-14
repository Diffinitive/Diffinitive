help:
	@echo 'Targets:'
	@echo '    help        - Show this help.'
	@echo '    docs        - Generate docs for webserver deployment.'
	@echo '    localdocs   - Generate docs for local viewing.'

docs:
	julia --project=docs --startup-file=no docs/make.jl --prettyurls

localdocs:
	julia --project=docs --startup-file=no docs/make.jl

clean:
	rm -r docs/build

.PHONY: help docs docs-local clean
