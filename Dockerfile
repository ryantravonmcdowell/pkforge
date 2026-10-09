FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[anthropic]"

# Run as an unprivileged user; outputs go to /work (mount a volume there to keep them).
RUN useradd --create-home --uid 1000 analyst && mkdir /work && chown analyst /work
USER analyst
WORKDIR /work

ENTRYPOINT ["pkforge"]
CMD ["demo", "--out-dir", "/work/out"]
