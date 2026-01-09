#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Stop and remove existing container if it exists
docker stop job-api 2>/dev/null
docker rm job-api 2>/dev/null

# Run the container
docker run -d \
  --name job-api \
  -p 80:8000 \
  -e AWS_REGION=us-east-1 \
  -v ${SCRIPT_DIR}/secrets/s3_bucket.txt:/run/secrets/s3_bucket:ro \
  -v ${SCRIPT_DIR}/secrets/job_queue_url.txt:/run/secrets/job_queue_url:ro \
  --restart unless-stopped \
  job-api

echo "Container started. Run ./logs.sh to view logs."