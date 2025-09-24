# FIRM_CE

- Installation instructions

- Data download link

- Basic configuration instructions

- Basic FIRM overview

- Broad optimum test

## Installation
For usage:

`pip install .`

For development:

`pip install -e .[dev]`

## Python Tools

These are the python tools conducting code checks.

| Name | Configuration file | Purpose | URL |
| --- | --- | --- | --- |
| flake8 | setup.cfg |  Linter | https://flake8.pycqa.org/en/latest/ |
| black | pyproject.toml | Formatter | https://github.com/psf/black |
| isort | pyproject.toml | Formats import statements | https://pypi.org/project/isort/ |
| safety | - | Checks for known security vulnerabilities in imported modules | https://pypi.org/project/safety/ |
| codespell | - | Spellchecker | https://pypi.org/project/codespell/ |
| bandit | pyproject.toml | Checks your code for security issues | https://github.com/PyCQA/bandit |
| pytest | pytest.ini | Testing Framework | https://docs.pytest.org/ |

Some tools have a configuration settings. The table above indicates in which configuration file to look for settings for that tool. For example, the flake8 settings can be found in the `[flake8]` section of setup.cfg. Others can be find in pyproject.toml

In order to get all the tools to work well together the chosen line length must be consistent. You will see
a value of 120 appearing in multiple places in `pyproject.toml`, `setup.cfg` and also in the editor settings, for example, settings.json for vscode.

We also recommend using [mypy](http://www.mypy-lang.org/). The `pyproject.toml` contains a default configuration and mypy can be enabled in your editors settings (see below)

## .gitignore

The .gitignore is based off [Github standard python .gitignore](https://github.com/github/gitignore/blob/main/Python.gitignore) with a few extra exclusions for files associated with code editors and operating systems.

### vscode

The file `vscode/settings.json` is an example configuration for vscode. To use these setting copy this file to `.vscode/settings,json`

The main features of this settings file are:
    - Enabling flake8 and disabling pylint
    - Autoformat on save (using the black and isort formatters)

Settings that you may want to change:
- Set the python path to your python in your venv with `python.defaultInterpreterPath`.
- Enable mypy by setting `python.linting.mypyEnabled` to true in settings.json.

