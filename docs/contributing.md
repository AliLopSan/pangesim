# Contributing to Pangesim

Thank you for your interest in contributing to `Pangesim`!
This page provides guidelines for contributing to the project, including how to set up your
development environment and submit changes.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating, you are expected to uphold this code.

In short:

- Be respectful and inclusive in all interactions.
- Harassment, discrimination, and personal attacks of any kind will not be tolerated.
- Focus feedback on ideas and code, not on individuals.

If you witness or experience unacceptable behavior, please report it by opening a GitHub issue or by contacting the maintainer directly.



## Development environment setup with uv

If you want to contribute, please use the package and project manager
[uv](https://docs.astral.sh/uv/). See [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.



## Code style and linting 

Please follow the [Google style guide](https://google.github.io/styleguide/pyguide.html) for Python code style and documentation.

Additionally, please adhere to the following guidelines:

- Keep line lengths to a maximum of **100** characters.
- Use type hints for all function arguments and return types.
- The `Returns` section in docstring should use an additional indentation level for text that does
  not fit in a single line to ensure proper formatting in the generated documentation. In case of
  multiple return values, start the description of each return value on a new line using the same
  indentation level as the first return value.
- f-strings should be used for string formatting whenever possible.

Make sure to write clear and concise docstrings for all functions, classes, and modules.



## Running Tests

The test suite uses [pytest](https://docs.pytest.org).

Make sure you have set up your development environment with `uv`.

Install the test dependencies and run the tests:

```bash
uv sync --group test
uv run pytest
```

This will run all the tests in the `tests/` directory.
