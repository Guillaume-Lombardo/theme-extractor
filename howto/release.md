# How To Release to PyPI

## Goal
Publish `theme-extractor` distributions to PyPI (or TestPyPI) from GitHub Actions.

## Prerequisites
- Repository workflow permissions enabled for OIDC.
- Trusted Publisher configured on PyPI for this GitHub repository and workflow.
- Project version updated in `pyproject.toml` before release.

## Release Workflow
The repository provides `.github/workflows/release.yml`.

It supports:
- automatic publish to PyPI when a GitHub Release is published
- manual publish to TestPyPI/PyPI via `workflow_dispatch`

## Recommended Sequence
1. Update version in `pyproject.toml`.
2. Merge changes to `main`.
3. Create and publish a GitHub Release with tag `v<version>`.
4. Workflow validates `tag == project.version`, builds `sdist` + wheel, runs `twine check`, then publishes.

## Manual Dry-Run to TestPyPI
From GitHub Actions:
- run workflow `Release`
- choose `publish_target=testpypi`

## Troubleshooting
- Tag/version mismatch:
  - ensure `pyproject.toml` version matches release tag without `v` prefix.
- Trusted publishing failure:
  - verify PyPI trusted publisher settings (repo, workflow name/path, environment).
- Metadata/build failure:
  - run locally:

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
```
