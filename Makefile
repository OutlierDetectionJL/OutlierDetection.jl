# Test
test:
	julia --project test/runtests.jl

# Documentation
# Make sure that that the `mkdocs` command is available and `mkdocs-material` is installed
docs_markdown:
	julia --project=docs/ docs/make.jl
docs_html:
	mkdocs build --clean --config-file=docs/mkdocs.yml
docs_serve:
	mkdocs serve --config-file=docs/mkdocs.yml
docs_build: docs_markdown docs_html
docs: docs_markdown docs_html docs_serve

# Benchmark
benchmark:
	julia --project=benchmark/ benchmark/runbenchmarks.jl
