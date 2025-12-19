import torch
import torch.nn as nn
from features import FeatureExtractor

class SimpleHybridDetector(nn.Module):

    def __init__(self, input_dim=775, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # Feature extractor (frozen)
        self.feature_extractor = FeatureExtractor()
        
        # Simple MLP classifier (trainable)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 2 classes: real (0), fake (1)
        )
    
    def forward(self, image_paths):
        # Extract features for each image
        features = []
        for path in image_paths:
            feat = self.feature_extractor.extract_all_features(path)
            features.append(feat)
        
        features = torch.stack(features)
        
        features = features.to(next(self.parameters()).device)
        # Classify
        logits = self.classifier(features)
        return logits
    
    def predict(self, image_path):
        self.eval()
        with torch.no_grad():
            logits = self.forward([image_path])
            probs = torch.softmax(logits, dim=1)[0]
            
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()
            
            return {
                'prediction': 'fake' if pred_class == 1 else 'real',
                'confidence': confidence,
                'probabilities': {
                    'real': probs[0].item(),
                    'fake': probs[1].item()
                }
            }
    
    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)