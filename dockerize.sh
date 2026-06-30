tugboat create \
  -e 'data/*' \
  -e '!data/cleaned-data_2023-03-28.csv' \
  -e .binder/ \
  -e .claude/ \
  -e .pytest_cache/ \
  -e .venv/ \
  -e 'figures/*' \
  -e '!figures/.gitignore' \
  -e resources/ \
  -e CLAUDE.md \
  --no-detect-r \
  --verbose

tugboat binderize --no-detect-r -b "main"