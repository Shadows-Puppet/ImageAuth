# AI-Generated Image Detection Tool

![Coverage](https://codecov.io/gh/Shadows-Puppet/ImageAuth/branch/main/graph/badge.svg)

## Overview
This project is a web-based tool for detecting AI-generated images using a **hybrid detection pipeline** that combines semantic, statistical, and signal-level analysis techniques. The goal is to improve robustness and generalization by avoiding reliance on a single detection method.

The system consists of a **React frontend**, a **FastAPI backend**, and multiple image analysis modules, hosted on **AWS** and can be accessed at [https://imageauth.dev](https://imageauth.dev).

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
FastAPI Server (AWS EC2)
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

**Infrastructure / Deployment**
- AWS EC2 (server)
- AWS S3 (image storage)
- AWS SQS (job scheduling)
- Local GPU worker (inference)
- Docker (portability and scalability)

**Planned**
- ZED (Zero-shot Entropy-based Detection)

---

## Disclaimer
This tool is intended for research and educational purposes. AI-generated image detection is an evolving problem, and results should not be treated as definitive proof of authenticity.

---

## License
MIT License
