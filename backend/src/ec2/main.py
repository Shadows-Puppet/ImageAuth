from fastapi import FastAPI, Request
import json
import boto3
from botocore.exceptions import ClientError
import uuid
import os

app = FastAPI()

# Get configuration from environment
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET = os.getenv("S3_BUCKET")
JOB_QUEUE_URL = os.getenv("JOB_QUEUE_URL")

# Initialize clients as None
s3_client = None
sqs_client = None

def get_s3_client():
    global s3_client
    if s3_client is None:
        s3_client = boto3.client("s3", region_name=AWS_REGION)
    return s3_client

def get_sqs_client():
    global sqs_client
    if sqs_client is None:
        sqs_client = boto3.client("sqs", region_name=AWS_REGION)
    return sqs_client

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/upload-url")
def upload_url(filename: str):
    job_id = str(uuid.uuid4())
    key = f"uploads/{job_id}_{filename}"
    url = get_s3_client().generate_presigned_url(
        "put_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=3600
    )
    return {"upload_url": url, "s3_key": key, "job_id": job_id}

@app.post("/submit-job")
async def submit_job(request: Request):
    data = await request.json()
    s3_key = data["s3_key"]
    job_id = data["job_id"]
    
    get_sqs_client().send_message(
        QueueUrl=JOB_QUEUE_URL,
        MessageBody=json.dumps({"image_s3_key": s3_key, "job_id": job_id})
    )
    return {"job_id": job_id, "message": "Job submitted successfully"}

@app.get("/results/{job_id}")
def results(job_id: str):
    key = f"results/{job_id}.json"
    try:
        obj = get_s3_client().get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(obj["Body"].read())
        return {"job_id": job_id, "result": data}
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return {"job_id": job_id, "result": "pending"}
        raise