FROM node:22-slim AS ui
WORKDIR /ui
COPY ui/package*.json ./
RUN npm install
COPY ui/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .
COPY configs ./configs
COPY --from=ui /ui/dist ./ui/dist
ENV AML_TRIAGE_REPORTS=/app/reports AML_TRIAGE_ARTIFACTS=/app/artifacts
EXPOSE 8000
CMD ["aml-triage", "serve", "--host", "0.0.0.0", "--port", "8000"]
