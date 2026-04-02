# Docker Setup Guide - Crypto Intelligence Terminal

## Prerequisites
- Docker Desktop installed (Windows/Mac) or Docker Engine (Linux)
- Docker Compose installed
- Your crypto intelligence terminal app

---

## Step-by-Step Setup

### Step 1: Create .env File
Create a `.env` file in your project root with environment variables:

```env
# Database Configuration
DB_USER=crypto_admin
DB_PASSWORD=secure_password_here
DB_NAME=crypto_intelligence

# API Keys (optional - add if you have them)
NEWS_API_KEY=your_newsapi_key
BINANCE_API_KEY=your_binance_key
COINDESK_API_KEY=your_coindesk_key

# Streamlit Config
STREAMLIT_THEME=dark
PYTHONUNBUFFERED=1
```

**Important:** Never commit .env file to git. Add to .gitignore:
```
.env
.env.local
```

---

### Step 2: Update Dockerfile (Current Setup is Good)

Your Dockerfile is ready. It includes:
- ✅ Python 3.9 slim base image
- ✅ System dependencies for ML/DB
- ✅ Port 8501 exposed for Streamlit
- ✅ Auto-runs terminal.py on startup

To also run WebSocket server, update the Dockerfile CMD:

```dockerfile
# Option 1: Run with startup script
CMD ["sh", "-c", "python websocket_server.py & streamlit run terminal.py --server.port=8501 --server.address=0.0.0.0"]

# Option 2: Use supervisord (more robust)
# Install supervisor and use supervisord config
```

---

### Step 3: Update docker-compose.yml

Your docker-compose.yml is good! It includes:
- ✅ PostgreSQL database service
- ✅ Streamlit app service
- ✅ Volume persistence for database
- ✅ Environment variable loading from .env

**Enhanced version with WebSocket server:**

```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    container_name: crypto_db
    env_file:
      - .env
    environment:
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=${DB_NAME}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - crypto_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  websocket_server:
    build: .
    container_name: crypto_websocket
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
    networks:
      - crypto_network
    command: python websocket_server.py
    restart: unless-stopped

  app:
    build: .
    container_name: crypto_app
    ports:
      - "8501:8501"
    env_file:
      - .env
    depends_on:
      - db
      - websocket_server
    extra_hosts:
      - "host.docker.internal:host-gateway"
    networks:
      - crypto_network
    environment:
      - DB_HOST=db
      - DB_PORT=5432
      - WEBSOCKET_URL=http://websocket_server:8000
    restart: unless-stopped

networks:
  crypto_network:
    driver: bridge

volumes:
  postgres_data:
```

---

## Building and Running

### Option 1: Build and Run with Docker Compose (Recommended)

```bash
# Navigate to project directory
cd d:\Hackathon\crypto-intelligence-terminal-main

# Build all images
docker-compose build

# Run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Build and Run Individual Containers

```bash
# Build the image
docker build -t crypto-intelligence-terminal:latest .

# Run Streamlit app
docker run -d \
  --name crypto_app \
  -p 8501:8501 \
  --env-file .env \
  crypto-intelligence-terminal:latest

# Run WebSocket server
docker run -d \
  --name crypto_websocket \
  -p 8000:8000 \
  --env-file .env \
  crypto-intelligence-terminal:latest \
  python websocket_server.py
```

---

## Accessing Your App

Once running:
- **Streamlit Dashboard**: http://localhost:8501
- **WebSocket Server**: ws://localhost:8000/ws
- **REST API**: http://localhost:8000/prices/{symbol}

---

## Docker Commands Reference

```bash
# View running containers
docker ps

# View all containers
docker ps -a

# View container logs
docker logs crypto_app
docker logs crypto_websocket
docker logs crypto_db

# Follow logs in real-time
docker logs -f crypto_app

# Stop specific container
docker stop crypto_app

# Remove container
docker rm crypto_app

# Execute command in running container
docker exec -it crypto_app bash

# Rebuild without cache
docker-compose build --no-cache

# Remove all stopped containers
docker container prune

# View container resource usage
docker stats

# Copy files to/from container
docker cp file.txt crypto_app:/app/
docker cp crypto_app:/app/file.txt ./
```

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8501
lsof -i :8501

# On Windows
netstat -ano | findstr :8501

# Kill process (get PID from above)
kill -9 <PID>  # Linux/Mac
taskkill /F /PID <PID>  # Windows
```

### Database Connection Issues
```bash
# Check if DB service is running
docker ps | grep crypto_db

# Verify database health
docker logs crypto_db

# Connect to database
docker exec -it crypto_db psql -U crypto_admin -d crypto_intelligence
```

### WebSocket Connection Issues
```bash
# Check WebSocket server logs
docker logs crypto_websocket

# Test WebSocket endpoint
curl http://localhost:8000/prices/BTC,ETH
```

### Rebuild Everything Fresh
```bash
# Stop and remove all
docker-compose down -v

# Remove images
docker rmi crypto-intelligence-terminal:latest

# Rebuild and start
docker-compose up -d --build
```

---

## Docker Best Practices

1. **Use .env files** - Never hardcode credentials
2. **Add health checks** - Ensure services are ready
3. **Use networks** - Better isolation and communication
4. **Volume mounting** - Persist important data
5. **Resource limits** - Prevent runaway containers
6. **Multi-stage builds** - Reduce image size
7. **Use specific tags** - Avoid `latest` in production

---

## Production Deployment

For production, consider:

```yaml
# Add resource limits
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

# Add SSL/TLS
  nginx:
    image: nginx:latest
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./ssl:/etc/nginx/ssl
```

---

## Next Steps

1. ✅ Create .env file with your configuration
2. ✅ Update docker-compose.yml with WebSocket service
3. ✅ Run `docker-compose up -d`
4. ✅ Access dashboard at http://localhost:8501
5. ✅ Monitor logs: `docker-compose logs -f`
6. ✅ Test WebSocket: `curl http://localhost:8000/prices/BTC`

---

## Support & Monitoring

```bash
# Monitor all services
docker stats

# Get container IP
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' crypto_app

# Check container environment
docker exec crypto_app env

# Verify mounted volumes
docker inspect crypto_app | grep -A 10 Mounts
```
