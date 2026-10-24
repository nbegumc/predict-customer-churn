.PHONY: setup hooks hooks-update data baseline push-utils clean lint requirements test mlflow-ui \
	serve docker-build up down release-bundle

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

test:
	uv run pytest

# browse the tracking store. SQLite rather than ./mlruns because the model
# registry needs a database-backed backend; see docs/adr/0001-mlflow.md
mlflow-ui:
	uv run mlflow ui --backend-store-uri $${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db} --port 5000

# the API on the host, reloading on edit
serve:
	uv run uvicorn src.api:app --reload --port 8000

# MODEL_RUN_ID comes from .env (see .env.example)
docker-build:
	docker build --build-arg MODEL_RUN_ID=$(MODEL_RUN_ID) -t churn-api:local .

up:
	docker compose up -d --build

down:
	docker compose down

# tar a bundle for a GitHub Release -- the model channel, since 30MB can
# never go in git. `make release-bundle RUN_ID=1024_141318_lgbm`
release-bundle:
	@test -n "$(RUN_ID)" || { echo "usage: make release-bundle RUN_ID=<run_id>"; exit 1; }
	@test -f models/$(RUN_ID)/manifest.json || { echo "no bundle at models/$(RUN_ID)"; exit 1; }
	mkdir -p dist
	tar -czf dist/bundle-$(RUN_ID).tar.gz -C models $(RUN_ID)
	@echo "dist/bundle-$(RUN_ID).tar.gz -- attach with: gh release create <tag> dist/bundle-$(RUN_ID).tar.gz"

# run every pre-commit hook without committing
lint:
	uv run pre-commit run --all-files

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .ipynb_checkpoints
