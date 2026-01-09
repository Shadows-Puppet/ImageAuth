# EC2 Backend API

FastAPI service for job processing with S3 and SQS integration. Runs in Docker on EC2.

## Quick Start

1. **Setup secrets**:
   ```bash
   mkdir -p secrets
   echo "your-bucket-name" > secrets/s3_bucket.txt
   echo "https://sqs.region.amazonaws.com/account-id/queue-name" > secrets/job_queue_url.txt
   ```

2. **Build and run**:
   ```bash
   chmod +x *.sh
   ./build.sh
   ./run.sh
   ```

3. **Check health**:
   ```bash
   curl http://localhost/health
   ```

## API Endpoints

- `GET /health` - Health check
- `GET /upload-url?filename=file.jpg` - Get presigned S3 upload URL
- `POST /submit-job` - Submit job to SQS queue
- `GET /results/{job_id}` - Get job results

## Management

```bash
./logs.sh      # View logs
./stop.sh      # Stop container
./restart.sh   # Rebuild and restart
```

## Requirements

- Docker on EC2
- EC2 IAM role with S3 and SQS permissions
- S3 bucket and SQS queue configured