FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JOURNAL_DATA_DIR=/data

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY journal_ai ./journal_ai
COPY data ./data

RUN useradd --create-home --uid 10001 journal && mkdir -p /data && chown -R journal /data
USER journal

EXPOSE 8000 8765
# Same image, two roles: the API/orchestrator (default) or the MCP tool server:
#   docker run ... python -m journal_ai.mcp_server --port 8765
CMD ["uvicorn", "journal_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
