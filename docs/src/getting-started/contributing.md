# How to Contribute

OutlierDetection.jl is a community-driven project and your help is extremely welcome. If you get stuck, please don't hesitate to [chat with us](https://discord.gg/5ErtExMV) or [raise an issue](https://github.com/davnn/OutlierDetection.jl/issues/new/choose). Take a look at Github's [How to Contribute Guide](https://opensource.guide/how-to-contribute/) to find out more about what it means to contribute.

**Note:** To avoid duplicating work, it is highly advised that you search through the [issue tracker](https://github.com/davnn/OutlierDetection.jl/issues) and the [PR list](https://github.com/davnn/OutlierDetection.jl/pulls). If in doubt about duplicated work, or if you want to work on a non-trivial feature, it’s recommended to first open an issue in the issue tracker to get some feedbacks from core developers.

## Areas of contribution

We value all kinds of contributions - not just code. The following table gives an overview of key contribution areas.

| Area               | Description                                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Documentation      | Improve or add docstrings, glossary terms, the user guide, and the example notebooks                                  |
| Testing            | Report bugs, improve or add unit tests, conduct field testing on real-world data sets                                 |
| Code               | Improve or add functionality, fix bugs                                                                                |
| Mentoring          | Onboarding and mentoring of new contributors                                                                          |
| Outreach           | Organize talks, tutorials or workshops, write blog posts                                                              |
| Maintenance        | Manage and review issues/pull requests |
| API design         | Design interfaces for detectors and other functionality                                                              |

Reporting bugs
--------------

We use GitHub issues to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the following rules before submitting:

- Verify that your issue is not being currently addressed by other [issues](https://github.com/davnn/OutlierDetection.jl/issues) or [pull requests](https://github.com/davnn/OutlierDetection.jl/pulls).
- Please ensure all code snippets and error messages are formatted in appropriate code blocks. See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).
- Please be specific about what detectors and/or functions are involved and the shape of the data, as appropriate; please include a [reproducible](https://stackoverflow.com/help/mcve) code snippet or link to a [gist](https://gist.github.com). If an exception is raised, please provide the traceback.

## The contribution workflow

The preferred workflow for contributing to OutlierDetection's repository is to fork the [main repository](https://github.com/davnn/OutlierDetection.jl) on GitHub, clone, and develop on a new branch.

1. Fork the [project repository](https://github.com/davnn/OutlierDetection.jl) by clicking on the \'Fork\' button near the top right of the page. This creates a copy of the code under your GitHub user account. For more details on how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. [Clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) your fork of the OutlierDetection.jl repo from your GitHub account to your local disk:

```bash
git clone git@github.com:USERNAME/OutlierDetection.jl.git
cd OutlierDetection.jl
```

3. Configure and link the remote for your fork to the upstream repository:

```bash
git remote -v
git remote add upstream https://github.com/davnn/OutlierDetection.git
```

4. Verify the new upstream repository you\'ve specified for your fork:

```bash
git remote -v
> origin    https://github.com/USERNAME/YOUR_FORK.git (fetch)
> origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
> upstream  https://github.com/davnn/OutlierDetection.jl.git (fetch)
> upstream  https://github.com/davnn/OutlierDetection.jl.git (push)
```

5. [Sync](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork) the `main` branch of your fork with the upstream repository:

```bash
git fetch upstream
git checkout main --track origin/main
git merge upstream/main
```

6. Create a new `feature` branch from the `main` branch to hold your changes:

```bash
git checkout main
git checkout -b <my-feature-branch>
```

Always use a `feature` branch. It\'s good practice to never work on the `main` branch! Name the `feature` branch after your contribution.

7. Develop your contribution on your feature branch. Add changed files using `git add` and then `git commit` files to record your changes in Git:

```bash
git add <modified_files>
git commit
```

8. When finished, push the changes to your GitHub account with:

```bash
git push --set-upstream origin my-feature-branch
```

9. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork) to create a pull request from your fork. If your work is still work in progress, you can open a draft pull request. We recommend to open a pull request early, so that other contributors become aware of your work and can give you feedback early on.

10. To add more changes, simply repeat steps 7 - 8. Pull requests are
 updated automatically if you push new changes to the same branch.

If any of the above seems like magic to you, please look up the [Git documentation](https://git-scm.com/documentation) on the web. If you get stuck, feel free to [chat with us](https://discord.gg/5ErtExMV).

## Continuous integration

We use continuous integration services on GitHub to automatically check if new pull requests do not break anything on all the Julia versions we support. The main quality control measures right now are [unit testing](#Unit-testing) and [test coverage](#Test-coverage). In the future we additionally want to check code style and formatting.

### Unit testing

We use Julia's built-in [Unit Testing](https://docs.julialang.org/en/v1/stdlib/Test/). The tests can be found in the [test](https://github.com/davnn/OutlierDetection.jl/tree/master/test) folder. To check if your code passes all tests make sure that you have the `OutlierDetection` environment activated and run `] test` from the Julia console.

### Test coverage

We use the [Coverage.jl](https://github.com/JuliaCI/Coverage.jl) package and [codecov](https://codecov.io) to measure and compare test coverage of our code.

## API design

We follow the general design approach chosen by MLJ, which is described in the paper ["Designing Machine Learning Toolboxes: Concepts, Principles and Patterns"](https://arxiv.org/abs/2101.04938). Additionally, we are always looking for feedback and improvement suggestions!

## Documentation

We use [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and [mkdocs](https://github.com/mkdocs/mkdocs/) to build and deploy our online documention.

The source files used to generate the online documentation can be found in [docs/src/](https://github.com/davnn/OutlierDetection.jl/tree/master/docs/src). For example, the main configuration file for mkdocs is [mkdocs.yml](https://github.com/davnn/OutlierDetection.jl/tree/master/docs/src/mkdocs.yml) and the main page is [index.md](https://github.com/davnn/OutlierDetection.jl/tree/master/docs/src/index.md). To add new pages, you need to add a new `.md` file and include it in the `mkdocs.yml` file.

To build the documentation locally, you need to navigate to [docs/](https://github.com/davnn/OutlierDetection.jl/tree/master/docs) and

1. Build the markdown files with Documenter.jl:

```bash
julia --project make.jl
```

2. To build the website using the markdown files, run:

```bash
mkdocs build # optionally run `mkdocs serve` to build and serve locally
```

You can find the generated files in the `OutlierDetection.jl/docs/site/` folder. To view the website, open `OutlierDetection.jl/docs/site/index.html` with your preferred web browser or use `mkdocs serve` to start a local documentation server.

## Coding style

We use [DocumentFormat.jl](https://github.com/julia-vscode/DocumentFormat.jl) as a code formatter. Additionally, we use a maximum line length of 120 characters.

## Acknowledging contributions

We follow the [all-contributors specification](https://allcontributors.org) and recognise various types of contributions. Take a look at our past and current [contributors](https://github.com/davnn/OutlierDetection.jl/blob/main/CONTRIBUTORS.md)! If you are a new contributor, please make sure we add you to our list of contributors. All contributions are recorded in [.all-contributorsrc](https://github.com/davnn/OutlierDetection.jl/blob/main/.all-contributorsrc).
