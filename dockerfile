# Use official Python image
FROM python:3.12-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y curl

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only necessary files first
COPY pyproject.toml poetry.lock* README.md ./

COPY  poetry.lock* ./
COPY .env .env
# Install Python dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi


# Copy rest of the app
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
