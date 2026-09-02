.PHONY: setup hooks hooks-update data baseline blend push-utils clean lint requirements

# first-time setup: environment + git hooks
setup:
	uv sync
	$(MAKE) hooks

# install the pre-commit git hooks. Safe to re-run; needed once per clone,
# because .git/hooks is local and never travels with the repo.
hooks:
	@git rev-parse --git-dir >/dev/null 2>&1 \
		|| { echo "not a git repository — run 'git init' first"; exit 1; }
	uv run pre-commit install

# bump the pinned hook versions, then check nothing broke
hooks-update:
	uv run pre-commit autoupdate
	uv run pre-commit run --all-files

data:
	bash scripts/download_data.sh

baseline:
	uv run python -m src.train --model lgbm --notes "baseline"

# pip-installable requirements for places uv isn't available (e.g. Kaggle kernels)
requirements:
	uv export --no-hashes --no-annotate --no-emit-project --format requirements-txt -o requirements.txt

push-utils:
	uv run kaggle datasets version -p kaggle/utils -m "sync src/"

# run every pre-commit hook without committing
lint:
	uv run pre-commit run --all-files

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .ipynb_checkpoints
