# Test
test:
	julia --project test/runtests.jl

# Documentation
# Make sure that that the `mkdocs` command is available and `mkdocs-material` is installed
docs_md:
	julia --project=docs/ docs/make.jl
docs_html: docs_md
	mkdocs build --clean --config-file=docs/mkdocs.yml
docs_serve: docs_md
	mkdocs serve --config-file=docs/mkdocs.yml

# Benchmark
benchmark:
	julia --project=benchmark/ benchmark/runbenchmarks.jl
