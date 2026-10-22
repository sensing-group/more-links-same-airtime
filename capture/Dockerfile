# Stage 1: Build frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.12-slim
WORKDIR /app

# Install backend dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install opencsi from vendored copy
COPY opencsi/ /tmp/opencsi/
RUN pip install --no-cache-dir /tmp/opencsi/ && rm -rf /tmp/opencsi/

# Copy backend code
COPY backend/ ./

# Copy built frontend to static/
COPY --from=frontend-build /app/frontend/dist ./static/

EXPOSE 8000

CMD ["python", "main.py"]
