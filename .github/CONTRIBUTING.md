
# Contributing to Statwrap

This documentation covers the contribution process for `statwrap`. Contributions are welcome, and we appreciate your interest in making `statwrap` better!

## Table of Contents
1. [Introduction](#introduction)
2. [How to Contribute](#how-to-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Submitting Pull Requests](#submitting-pull-requests)
3. [Development Process](#development-process)
    - [Setting Up Your Development Environment](#setting-up-your-development-environment)
    - [Coding Standards](#coding-standards)
    - [Testing](#testing)
4. [Style Guides](#style-guides)
    - [Git Commit Messages](#git-commit-messages)
    - [Code Style](#code-style)
5. [Documentation](#documentation)
    - [Building the Documentation](#building-the-documentation)
    - [Documenting Code](#documenting-code)
6. [License](#license)

## Introduction

Thank you for considering contributing to `statwrap`. This guide outlines the process and expectations for contributing to the project, whether you're reporting a bug, suggesting an enhancement, or submitting a pull request.

## How to Contribute

### Reporting Bugs

- **Where to Report:** Please report bugs via the [GitHub Issues](link-to-issues) page.
- **How to Write a Good Bug Report:**
  - Provide a clear and concise description of the issue.
  - Include steps to reproduce the bug, expected behavior, and what actually happened.
  - Attach relevant code snippets or screenshots if necessary.
  - Label as Bug

### Suggesting Enhancements

- **Where to Suggest:** Enhancement requests should be submitted through [GitHub Issues](link-to-issues).
- **How to Write a Good Enhancement Suggestion:**
  - Clearly describe the enhancement and the problem it solves.
  - Provide any relevant context, examples, or documentation.
  - Label as Enhancement

### Submitting Pull Requests

- **Pull Request Process:**
  1. Fork the repository and create a new branch for your changes.
  2. Make your changes, ensuring you follow the [Coding Standards](#coding-standards).
  3. Commit your changes with a descriptive message.
  4. Push your changes to your fork and submit a pull request to the main repository.

- **Guidelines for Pull Requests:**
  - Ensure your code passes all tests and follows the project's style guides.
  - Build the docs using Sphinx and attach a screenshot of the changed output.
  - Reference any relevant issues in your pull request description.
  - Keep your pull request focused on a single issue or feature.

## Development Process

### Setting Up Your Development Environment

- **Prerequisites:**
  - Python 3.8 or later.
  - Required packages:
    - `pandas`
    - `numpy`
    - `scipy`
    - `IPython`
    - `matplotlib`
    - `statsmodels`
    - `ipywidgets`
    - `odfpy`
    - `openpyxl`

- **Installation Steps:**
  1. Fork and clone the repository:
     ```bash
     git clone https://github.com/your-username/statwrap.git
     cd statwrap
     ```
  2. Install the required dependencies using `pip`:
     ```bash
     pip install -r requirements.txt
     ```

### Coding Standards

- **Language-Specific Guidelines:** Follow PEP 8 for Python code.
- **Code Reviews:** All code changes must go through a review process. Be responsive to feedback and ready to make necessary changes.

### Testing

- **Testing Framework:** The package uses `unittest` for testing. Ensure all new features or bug fixes include corresponding test cases.
- **Test Files:** Test cases should be placed in the `tests` directory and follow the structure and naming conventions of existing tests.
- **Running Tests:** Run the test suite using:
  ```bash
  python -m unittest discover tests
  ```
- **Test Example:**
  ```python
  import unittest
  from statwrap.fpp import r

  class TestCorrelation(unittest.TestCase):

      def setUp(self):
          self.x = [1, 2, 3]
          self.y = [4, 5, 6]

      def test_correlation(self):
          result = r(self.x, self.y)
          expected = 1.0
          self.assertAlmostEqual(result, expected)

  if __name__ == "__main__":
      unittest.main()
  ```

## Style Guides

### Git Commit Messages

- **Format:** Use clear, descriptive commit messages. Follow the format:
  ```
  ENH: Brief description of the enhancement
  ```
- **Best Practices:** 
  - Start with a capital letter.
  - Keep the message concise but informative.
  - If responding to an issue (BUG or ENH), include 'fixes #{issue number}' in the commit message.

### Code Style

- **Language-Specific Conventions:** Adhere to PEP 8 for Python. Use type hints where applicable.
- **Linting and Formatting:** Use `black` for code formatting and `flake8` for linting.

## Documentation

### Building the Documentation

- **Sphinx Setup:** The documentation is built using Sphinx. Ensure you have Sphinx and the necessary extensions installed:
  ```bash
  pip install sphinx sphinx_rtd_theme
  ```
- **Building the Docs:**
  1. Navigate to the `docs` directory:
     ```bash
     cd docs
     ```
  2. Build the HTML documentation:
     ```bash
     make html
     ```
  3. The generated documentation can be found in `docs/_build/html/index.html`.

### Documenting Code

- **Docstring Conventions:** Follow the NumPy/SciPy documentation style for writing docstrings. This includes sections such as Parameters, Returns, Examples, and Notes.
- **API Documentation:** Ensure that all public functions and classes are well-documented. Private methods can be documented at your discretion.
- **Rebuilding Docs:** After making changes to the code or documentation, rebuild the documentation to ensure it is up-to-date.
- **Curent Documentation:** Found on this site: [StatWrap Documentation](https://statwrap.readthedocs.io/en/latest/)

## License

`statwrap` is licensed under the [BSD 3-Clause License](../LICENSE).
