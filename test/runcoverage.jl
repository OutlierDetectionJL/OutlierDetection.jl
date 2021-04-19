using Pkg
using Coverage

Pkg.test(; coverage=true)
coverage = process_folder()
LCOV.writefile("lcov.info", coverage)
