#!/bin/bash
set -e

# Create required directories
mkdir -p /var/log/nginx
touch /var/log/nginx/error.log /var/log/nginx/access.log
chown -R root:root /var/log/nginx

# Debug: Check frontend build exists
echo "Checking frontend build..."
ls -la /frontend/dist/

# Ensure nginx can access frontend files
chmod -R 755 /frontend/dist

# Start backend (FastAPI)
cd /app
uvicorn server:app --host 0.0.0.0 --port 9000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Debug: Test nginx config
echo "Testing nginx config..."
nginx -t

# Start nginx in foreground
echo "Starting nginx..."
nginx -g 'daemon off;' &
NGINX_PID=$!

# Monitor both processes
while kill -0 $BACKEND_PID && kill -0 $NGINX_PID > /dev/null 2>&1; do
    sleep 1
done

# If either process dies, kill the other and exit
kill -TERM $BACKEND_PID 2>/dev/null || true
kill -TERM $NGINX_PID 2>/dev/null || true
exit 1