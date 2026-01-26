#!/bin/bash
set -e

# Set AWS region directly
export AWS_REGION=us-east-2

# Read Docker secrets and export as environment variables
if [ -f /run/secrets/s3_bucket ]; then
    export S3_BUCKET=$(cat /run/secrets/s3_bucket)
fi

if [ -f /run/secrets/job_queue_url ]; then
    export JOB_QUEUE_URL=$(cat /run/secrets/job_queue_url)
fi

# Debug: print environment variables (remove after testing)
echo "AWS_REGION: $AWS_REGION"
echo "S3_BUCKET: $S3_BUCKET"
echo "JOB_QUEUE_URL: $JOB_QUEUE_URL"

# Start the application
exec uvicorn main:app --host 0.0.0.0 --port 8000