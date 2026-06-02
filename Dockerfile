FROM buildpack-deps:trixie
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/
COPY . /av-optimal-policy
WORKDIR /av-optimal-policy
RUN uv sync