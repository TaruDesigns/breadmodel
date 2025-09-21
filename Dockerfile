# Using ultralytics without GPU
FROM ultralytics/ultralytics:latest-cpu

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ADD . /app
WORKDIR /app

RUN uv pip compile pyproject.toml --output-file requirements.txt
RUN uv pip install -r requirements.txt --system --break-system-packages
ENV PYTHONPATH="/app/src"

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]