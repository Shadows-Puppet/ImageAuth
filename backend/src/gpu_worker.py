import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
import json
import time
from PIL import Image
import io
from typing import Dict
from features import FeatureExtractor
from train import Classifier
import torch

load_dotenv()
PROFILE = os.getenv("PROFILE")
REGION = "us-east-2"
S3_BUCKET = os.getenv("S3")
QUEUE_URL =  os.getenv("JOB_QUEUE_URL")

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
s3 = session.client("s3")
sqs = session.client("sqs")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
extractor = None

def model_load():
    global model, extractor
    
    print("Loading model...")
    
    # Load classifier
    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
    model = Classifier(input_dim=775)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load feature extractor
    extractor = FeatureExtractor(device=device)
    extractor.load_normalizers("checkpoints/normalizers.npz")
    
    print(f"✓ Model loaded on {device}")

def test_s3():
    """Test S3 connectivity"""
    print("=== Testing S3 ===")
    try:
        resp = s3.list_buckets()
        print("Buckets:", [b["Name"] for b in resp["Buckets"]])
        
        # Try to list objects in the working bucket
        objects = s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=5)
        if "Contents" in objects:
            print(f"Sample objects in {S3_BUCKET}:")
            for obj in objects["Contents"]:
                print(f"  - {obj['Key']}")
        else:
            print(f"No objects in {S3_BUCKET}")
            
    except ClientError as e:
        print("❌ Failed S3 test:", e)

def test_sqs():
    """Test SQS connectivity"""
    print("\n=== Testing SQS ===")
    try:
        resp = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=1
        )
        
        if "Messages" not in resp:
            print("✓ Queue reachable, no messages available")
        else:
            print(f"✓ Queue reachable, {len(resp['Messages'])} message(s) available")
            
    except ClientError as e:
        print("❌ Failed SQS test:", e)

        
def extract_features_from_image(image: Image.Image) -> torch.Tensor:
    """Extract features from PIL Image"""
    
    # Resize to 256x256
    image = image.resize((256, 256), Image.BILINEAR)
    
    # Extract CLIP features
    clip_feat = extractor.extract_clip_features(image)
    
    # Extract frequency features
    freq_feat = extractor.extract_frequency_features(image)
    freq_feat = torch.from_numpy(freq_feat).float()
    
    # Extract compression features
    comp_feat = extractor.extract_compression_features(image)
    comp_feat = torch.from_numpy(comp_feat).float()
    
    # Normalize
    if extractor.is_fitted:
        freq_feat = (freq_feat - torch.from_numpy(extractor.freq_mean).float()) / \
                    torch.from_numpy(extractor.freq_std).float()
        comp_feat = (comp_feat - torch.from_numpy(extractor.comp_mean).float()) / \
                    torch.from_numpy(extractor.comp_std).float()
    
    # Combine all features
    features = torch.cat([clip_feat, freq_feat, comp_feat])
    
    return features

def process_image(image_data):
    # Read image
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Extract features
    features = extract_features_from_image(image)
    features = features.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)[0]
        
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
    
    # RETURN THE RESULT
    return {
        "predicted_class": pred_class,
        "confidence": float(confidence),
        "all_probabilities": {
            i: float(prob) for i, prob in enumerate(probs.tolist())
        }
    }

def process_job(message_body):
    try:
        job_data = json.loads(message_body)
        image_s3_key = job_data["image_s3_key"]
        job_id = job_data["job_id"]
        
        print(f"Processing job {job_id}")
        print(f"Downloading image from s3://{S3_BUCKET}/{image_s3_key}")
        
        # Download the image from S3
        response = s3.get_object(Bucket=S3_BUCKET, Key=image_s3_key)
        image_data = response["Body"].read()
        
        # Process it
        result = process_image(image_data)
        
        # Upload result
        result_key = f"results/{job_id}.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=result_key,
            Body=json.dumps(result),
            ContentType="application/json"
        )
        
        print(f"Result uploaded to s3://{S3_BUCKET}/{result_key}")
        return True
        
    except Exception as e:
        print(f"Error processing job: {e}")
        import traceback
        traceback.print_exc()  # This will show you the full error
        return False


def worker_loop():
    print("Worker started. Waiting for jobs...")

    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=300
            )

            if "Messages" not in resp:
                continue  # no jobs

            msg = resp["Messages"][0]
            receipt = msg["ReceiptHandle"]

            print(f"\nReceived message: {msg['MessageId']}")
                
            # Process the job
            success = process_job(msg["Body"])
            
            if success:
                # Delete message from queue if processing succeeded
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=receipt
                )
                print("Message deleted from queue")
            else:
                print("Processing failed, message will return to queue")
        
        except KeyboardInterrupt:
            print("\nShutting down worker...")
            break
        except ClientError as e:
            print(f"AWS error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    test_s3()
    test_sqs()
    model_load()
    worker_loop()