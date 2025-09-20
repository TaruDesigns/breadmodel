# Using ultralytics without GPU
FROM ultralytics/ultralytics:latest-cpu

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD . /app
WORKDIR /app

RUN uv sync --locked
ENV PYTHONPATH="/app/src"


# Command to run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]