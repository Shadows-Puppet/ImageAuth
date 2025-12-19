# AI-Generated Image Detection Tool

## Overview
This project is a web-based tool for detecting AI-generated images using a **hybrid detection pipeline** that combines semantic, statistical, and signal-level analysis techniques. The goal is to improve robustness and generalization by avoiding reliance on a single detection method.

The system consists of a **React frontend**, a **FastAPI backend**, and multiple image analysis modules, with planned cloud deployment on **AWS**.

---

## Features
- **CLIP-based semantic detection**  
  Uses vision–language embeddings to identify inconsistencies common in AI-generated imagery.

- **Frequency-domain analysis**  
  Examines spectral artifacts introduced by generative models that are often invisible in the spatial domain.

- **Compression artifact analysis**  
  Analyzes JPEG compression behavior to detect irregularities typical of synthetic images.

- **Modular backend architecture**  
  Detection methods are isolated and composable, enabling easy experimentation and extension.

---

## Architecture
```

React (Vite + TypeScript)
↓
FastAPI Server
↓
Detection Pipeline
├─ CLIP-based analysis
├─ Frequency-domain analysis
├─ Compression artifact analysis
└─ (Planned) ZED-based detection

```

---

## Tech Stack
**Frontend**
- React
- TypeScript
- Vite

**Backend**
- Python
- FastAPI
- PyTorch
- CLIP

**Planned / Optional**
- ZED (Zero-shot Entropy-based Detection)
- AWS (EC2, S3, local GPU for training and inference)

---

## Deployment
Planned deployment targets AWS:
- **EC2** for backend inference (CPU or GPU depending on model configuration)
- **S3** for image storage
- **Docker**: Docker-based deployment for portability and scalability

---

## Disclaimer
This tool is intended for research and educational purposes. AI-generated image detection is an evolving problem, and results should not be treated as definitive proof of authenticity.

---

## License
MIT License
