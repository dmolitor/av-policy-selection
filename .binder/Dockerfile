FROM rocker/binder:4

ENV RETICULATE_PYTHON_ENV=/home/rstudio/.venv
ENV VIRTUAL_ENV=/home/rstudio/.venv

COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/
COPY --chown=${NB_USER} . /home/rstudio
WORKDIR /home/rstudio
RUN uv sync