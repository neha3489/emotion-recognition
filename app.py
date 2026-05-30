"""
Facial Emotion Recognition — Gradio Web App
============================================
ResNet18 fine-tuned on FER-2013 | 6 emotions | MediaPipe face detection
"""

import cv2
import numpy as np
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import mediapipe as mp

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "emotion_model_v5.pth"
IMG_SIZE    = 64
CLASS_NAMES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI_MAP   = {
    "angry": "😠", "fear": "😨", "happy": "😊",
    "neutral": "😐", "sad": "😢", "surprise": "😲"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()
print("Model ready!")

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── MediaPipe face detector ───────────────────────────────────────────────────
mp_face       = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ── Core inference ────────────────────────────────────────────────────────────
def detect_and_predict(image_rgb: np.ndarray):
    h, w     = image_rgb.shape[:2]
    results  = face_detector.process(image_rgb)
    annotated    = image_rgb.copy()
    face_results = []

    if not results.detections:
        return annotated, []

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        face_crop = image_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        pil_face     = Image.fromarray(face_crop)
        input_tensor = transform(pil_face).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        top_idx    = int(np.argmax(probs))
        emotion    = CLASS_NAMES[top_idx]
        confidence = float(probs[top_idx])

        face_results.append({
            "emotion": emotion, "confidence": confidence, "probs": probs.tolist()
        })

        color = (34, 197, 94) if confidence > 0.6 else (251, 191, 36)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{EMOJI_MAP[emotion]} {emotion} {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
        cv2.putText(annotated, label, (x1+4, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return annotated, face_results


def format_html(face_results):
    if not face_results:
        return "<p style='color:#888;padding:12px'>No face detected. Try a clearer photo.</p>"

    html = []
    for i, res in enumerate(face_results, 1):
        html.append(
            f"<div style='margin-bottom:16px'>"
            f"<b>Face {i}</b> — "
            f"<span style='font-size:1.3em'>{EMOJI_MAP[res['emotion']]}</span> "
            f"<b>{res['emotion'].upper()}</b> ({res['confidence']:.1%})<br><br>"
        )
        for idx in np.argsort(res["probs"])[::-1]:
            name  = CLASS_NAMES[idx]
            pct   = res["probs"][idx] * 100
            color = "#22c55e" if idx == np.argmax(res["probs"]) else "#60a5fa"
            html.append(
                f"<div style='display:flex;align-items:center;margin-bottom:5px'>"
                f"<span style='width:90px;font-size:13px'>{EMOJI_MAP[name]} {name}</span>"
                f"<div style='flex:1;background:#e5e7eb;border-radius:4px;height:18px;margin:0 8px'>"
                f"<div style='width:{pct:.1f}%;background:{color};height:100%;border-radius:4px'></div></div>"
                f"<span style='width:40px;text-align:right;font-size:13px'>{pct:.1f}%</span></div>"
            )
        html.append("</div>")
    return "".join(html)


# ── Handlers ──────────────────────────────────────────────────────────────────
def predict_image(img):
    if img is None:
        return None, "<p>Upload an image to analyse.</p>"
    img_rgb = np.array(img)
    if img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[:, :, :3]
    annotated, results = detect_and_predict(img_rgb)
    return Image.fromarray(annotated), format_html(results)


def predict_webcam(frame):
    if frame is None:
        return None, ""
    img_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated, results = detect_and_predict(img_rgb)
    return Image.fromarray(annotated), format_html(results)


# ── Interview Practice ────────────────────────────────────────────────────────
interview_log = []

def analyze_frame(frame):
    if frame is None:
        return None, "", ""
    img_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated, results = detect_and_predict(img_rgb)

    if results:
        interview_log.append(results[0]["emotion"])

    if interview_log:
        from collections import Counter
        counts = Counter(interview_log)
        total  = len(interview_log)
        tl = "<b>Emotion timeline</b><br>"
        for em, cnt in counts.most_common():
            pct   = cnt / total * 100
            color = "#22c55e" if em in ["happy", "neutral"] else "#f87171"
            tl += (
                f"<div style='display:flex;align-items:center;margin:3px 0'>"
                f"<span style='width:90px;font-size:12px'>{EMOJI_MAP[em]} {em}</span>"
                f"<div style='flex:1;background:#e5e7eb;border-radius:4px;height:14px;margin:0 6px'>"
                f"<div style='width:{pct:.0f}%;background:{color};height:100%;border-radius:4px'></div></div>"
                f"<span style='width:38px;font-size:12px'>{pct:.0f}%</span></div>"
            )
    else:
        tl = "<p style='color:#888'>Start recording to see your timeline.</p>"

    feedback = ""
    if len(interview_log) > 10:
        from collections import Counter
        counts   = Counter(interview_log)
        total    = len(interview_log)
        positive = (counts.get("happy", 0) + counts.get("neutral", 0)) / total
        negative = (counts.get("sad", 0) + counts.get("fear", 0) + counts.get("angry", 0)) / total
        if positive > 0.6:
            feedback = "✅ Looking confident! Great mix of happy and neutral."
        elif negative > 0.4:
            feedback = "⚠️ Try to relax — showing quite a bit of nervous/negative emotion."
        else:
            feedback = "🔄 Mixed. Aim for more calm and positive expressions."
    else:
        feedback = "Keep recording for feedback..."

    return Image.fromarray(annotated), tl, feedback


def reset_interview():
    global interview_log
    interview_log = []
    return None, "<p style='color:#888'>Session reset.</p>", ""


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Facial Emotion Recognition") as demo:
    gr.Markdown(
        "# 🎭 Facial Emotion Recognition\n"
        "**ResNet18 · FER-2013 · 6 emotions · MediaPipe face detection**  \n"
        "Test accuracy: **63.6%** (vs 16.7% random baseline)"
    )

    with gr.Tabs():

        with gr.TabItem("📷 Image Upload"):
            gr.Markdown("Upload any photo — detects all faces and classifies emotions.")
            with gr.Row():
                with gr.Column():
                    img_in  = gr.Image(label="Input", type="pil")
                    img_btn = gr.Button("Analyse", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="Result")
                    img_res = gr.HTML()
            img_btn.click(predict_image, inputs=img_in, outputs=[img_out, img_res])

        with gr.TabItem("🎥 Live Webcam"):
            gr.Markdown("Real-time emotion detection from your webcam.")
            with gr.Row():
                with gr.Column():
                    cam_in  = gr.Image(label="Webcam", sources=["webcam"], streaming=True)
                with gr.Column():
                    cam_out = gr.Image(label="Result")
                    cam_res = gr.HTML()
            cam_in.stream(predict_webcam, inputs=cam_in, outputs=[cam_out, cam_res])

        with gr.TabItem("🎤 Interview Practice"):
            gr.Markdown(
                "Practice looking confident on camera.  \n"
                "Aim for **happy** and **neutral** — avoid fear, sad, or angry."
            )
            with gr.Row():
                with gr.Column():
                    iv_cam   = gr.Image(label="Webcam", sources=["webcam"], streaming=True)
                    iv_reset = gr.Button("Reset Session", variant="secondary")
                with gr.Column():
                    iv_out      = gr.Image(label="Live Detection")
                    iv_timeline = gr.HTML()
                    iv_feedback = gr.Textbox(label="Feedback", interactive=False)
            iv_cam.stream(analyze_frame, inputs=iv_cam,
                          outputs=[iv_out, iv_timeline, iv_feedback])
            iv_reset.click(reset_interview,
                           outputs=[iv_out, iv_timeline, iv_feedback])

    gr.Markdown(
        "---\n"
        "Model: ResNet18 fine-tuned on FER-2013 · "
        "[GitHub](https://github.com/YOUR-USERNAME/emotion-recognition)"
    )

if __name__ == "__main__":
    demo.launch(share=True)