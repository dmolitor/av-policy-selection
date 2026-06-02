FROM rocker/binder:4

COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/
COPY --chown=${NB_USER} . /home/rstudio
WORKDIR /home/rstudio
USER root
RUN RUN sed -i "/^RETICULATE_PYTHON=/d;/^VIRTUAL_ENV=/d" /usr/local/lib/R/etc/Renviron.site && \
    printf "%s\n" \
      "RETICULATE_PYTHON=/home/rstudio/.venv/bin/python" \
      "VIRTUAL_ENV=/home/rstudio/.venv" \
      >> /usr/local/lib/R/etc/Renviron.site
USER ${NB_USER}
RUN uv sync