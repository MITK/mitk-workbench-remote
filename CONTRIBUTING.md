# Contributing to mitk-workbench-remote

Thanks for considering contributing to mitk-workbench-remote!

---

## Repository model

The authoritative repository is hosted on GitLab (`git.dkfz.de`) and is the
single source of truth. GitHub hosts a public mirror: its branches are pushed
from GitLab and are read-only there, but its Issues and Pull Requests are open
for collaboration.

As a contributor you work entirely on GitHub: file bug reports and feature
requests on GitHub Issues, and open pull requests against `develop`. Because the
branches are mirrored from GitLab, an approved pull request is integrated by a
maintainer on the GitLab side rather than through the GitHub "Merge" button; the
result then mirrors back to GitHub.

---

## Getting Started

1. Fork & clone the repository:
   ```
   https://github.com/MITK/mitk-workbench-remote
   ```
2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Linux/Mac
   .venv\Scripts\activate      # on Windows
   ```
3. Go to the root of the repository.
4. Install all dependencies in editable mode:

   ```bash
   pip install -e ".[dev,all]"
   ```

---

## Development Workflow

### Development

1. Create your feature branch:
   ```bash
   git checkout -b feature/short-description
   ```
2. Make your code changes.

Develop your idea or bug fix in your venv/repo clone.
If you have ideas to improve, fix, or otherwise advance mitk-workbench-remote, **please feel wholeheartedly invited and encouraged to
discuss the issues and ideas beforehand** in the GitHub issue tracker:
https://github.com/MITK/mitk-workbench-remote/issues

### Pull request preparation

1. Format & lint your code before committing:

   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   ```

2. Run type checks:

   ```bash
   mypy src/
   ```

3. Run tests:

   ```bash
   pytest tests/ -v
   ```

Ensure all tests pass. Integration tests (marked with `@pytest.mark.integration`) require a running MITK Workbench instance and can be skipped if one is not available, as long as your change does not impact integration behavior.

### Making a pull request

1. Push your branch to your fork:
   ```bash
   git push origin feature/short-description
   ```
2. Go to your fork on GitHub and open a pull request against `develop`.
3. Include the following information:
   - A short, descriptive title
   - What changes you made
   - Why these changes are necessary
   - Any relevant issue numbers (e.g., Closes #42)
4. Ensure your branch passes CI checks:
   - All tests pass
   - `ruff` formatting and lint checks pass
   - `mypy` type checks pass

5. Request a review from maintainers if necessary.
6. Once approved, a maintainer will merge your PR.
7. After merge, delete your feature branch locally and on your fork.

---

## Guidelines

- Follow the coding conventions described in `CLAUDE.md`.
- Follow [PEP 8](https://peps.python.org/pep-0008/) coding style, enforced via `ruff`.
- Add tests for new features or bug fixes.
- Update documentation when you change public APIs.
- Keep commits small and meaningful.
- All HTTP communication must go through `RestTransport` — do not add `requests` calls outside `transport.py`.
- New exceptions should fit into the existing hierarchy in `errors.py`.
- The `contract/openapi.json` is the pinned interface contract with the MITK C++ REST server — update it when the API changes.

---

## Reporting Issues

Please use the GitHub issue tracker for bugs and feature requests:
https://github.com/MITK/mitk-workbench-remote/issues

Include:

- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (Python version, OS, MITK Workbench version if applicable)

---

Happy hacking!
