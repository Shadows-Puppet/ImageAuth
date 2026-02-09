import json
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from backend.src.ec2 import main
from botocore.exceptions import ClientError

client = TestClient(main.app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("backend.src.ec2.main.get_s3_client")
def test_upload_url(mock_get_s3):
    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.return_value = "https://fake-url"
    mock_get_s3.return_value = mock_s3

    response = client.get("/upload-url?filename=test.png")
    data = response.json()

    assert response.status_code == 200
    assert "upload_url" in data
    assert "s3_key" in data
    assert "job_id" in data
    assert data["upload_url"] == "https://fake-url"

    mock_s3.generate_presigned_url.assert_called_once()

@patch("backend.src.ec2.main.get_sqs_client")
def test_submit_job(mock_get_sqs):
    mock_sqs = MagicMock()
    mock_get_sqs.return_value = mock_sqs

    payload = {
        "s3_key": "uploads/test.png",
        "job_id": "testjob"
    }

    response = client.post('/submit-job', json=payload)

    assert response.status_code == 200
    assert response.json()["job_id"] == "testjob"

    mock_sqs.send_message.assert_called_once()

@patch("backend.src.ec2.main.get_s3_client")
def test_result_success(mock_get_s3):
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps({"label": 1}))
    }
    mock_get_s3.return_value = mock_s3

    response = client.get("/results/testjob")

    assert response.status_code == 200
    assert response.json()["result"] == {"label": 1}

@patch("backend.src.ec2.main.get_s3_client")
def test_result_success(mock_get_s3):
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey"}},
        "GetObject"
    )
    mock_get_s3.return_value = mock_s3

    response = client.get("/results/testjob")

    assert response.status_code == 200
    assert response.json()["result"] == "pending"
