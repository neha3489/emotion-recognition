# Facial Emotion Recognition

Real-time facial emotion recognition using ResNet18 transfer learning, deployed as an interactive web app.

**Live Demo:** []

---

## Demo

>

---

## What it does

Detects faces in images or webcam video and classifies each face into one of 6 emotions — angry, fear, happy, neutral, sad, surprise — with confidence scores for all classes.

**Interview Practice Mode** tracks your emotions over a session and tells you how confident you appear on camera.

---

## Results

| | Accuracy |
|---|---|
| **This model (ResNet18 + Mixup + OneCycleLR)** | **65.93%** |
| v3 baseline (ResNet18 clean) | 63.62% |
| Original model (CNN from scratch) | 49.00% |
| Random baseline (6 classes) | 16.67% |
| FER-2013 SOTA | 73.70% |

### Per-class accuracy

| Emotion | Accuracy | Grade |
|---|---|---|
| 😊 Happy | 86.4% | ✅ |
| 😲 Surprise | 78.6% | ✅ |
| 😐 Neutral | 64.2% | ✅ |
| 😠 Angry | 58.5% | ✅ |
| 😢 Sad | 55.1% | ⚠️ |
| 😨 Fear | 42.4% | ⚠️ |

*Fear and sad are the hardest to separate — they share visual features (downturned mouth, droopy eyes). This is a known challenge in FER-2013 even at SOTA level.*

---

## Model architecture

```
Input (64×64 RGB)
    ↓
ResNet18 (pretrained on ImageNet)
  Phase 1: base frozen — head only trained (8 epochs, LR=3e-3)
  Phase 2: full fine-tune with Mixup + OneCycleLR (45 epochs, LR=1e-4)
    ↓
GlobalAvgPool → FC(512→256) → ReLU → Dropout(0.5) → FC(256→6)
```

**Key training decisions:**
- **Mixup augmentation** (α=0.2) — blends sad/fear image pairs to help the model learn smoother decision boundaries between similar emotions
- **Frozen BatchNorm** during Phase 2 — preserves ImageNet statistics, worth +1-2% accuracy
- **Discriminative learning rates** — backbone at 1e-4, head at 5×1e-4
- **Class weights** — balanced + extra boost for fear (1.3×) and sad (1.2×)
- **OneCycleLR** with 10% warmup — no LR restart spikes

---

## Dataset

FER-2013 from Kaggle — 28,273 training / 7,067 test images, 48×48 grayscale (resized to 64×64 RGB).
6 classes used (disgust excluded — only ~500 samples vs 8,000+ for happy).

---

## Project structure

```
emotion-recognition/
├── app.py                        # Gradio web app
├── fed_train_resnet18_improved.py # Training script (PyTorch)
├── requirements.txt              # Dependencies
└── README.md

# Note: emotion_model_v5.pth is hosted on Hugging Face Spaces
# (43MB — too large for GitHub)
```


---

## Deploy to Hugging Face Spaces (free)

1. Create account at huggingface.co
2. New Space → SDK: Gradio → Visibility: Public
3. Upload `app.py`, `requirements.txt`, `emotion_model_v5.pth`
4. Builds automatically in ~5 minutes — you get a permanent public URL

---

## Tech stack

Python · PyTorch · ResNet18 · MediaPipe · Gradio · OpenCV · scikit-learn · NumPy

---

## key points

> Fine-tuned ResNet18 on FER-2013 (28K images, 6 classes) using two-phase transfer learning with Mixup augmentation and OneCycleLR, achieving **65.93% test accuracy** (vs 16.7% random baseline and 49% original model); replaced Haar Cascade with MediaPipe for robust real-time face detection; deployed as a live Gradio web app with webcam inference and interview practice mode.
