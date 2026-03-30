FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy ONLY the package files
COPY pyproject.toml poetry.lock* /app/

RUN pip install poetry && poetry config virtualenvs.create false

# This will now be CACHED unless you change a library
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --no-root --only main

# Copy only the source code
COPY . /app/

# NOTICE: No truenorth-*.json files are copied here anymore
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]