FROM rocker/binder:4

COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/
COPY --chown=${NB_USER} . /home/rstudio
WORKDIR /home/rstudio
RUN uv sync