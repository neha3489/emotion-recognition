"""
Facial Emotion Detection - v5 (ResNet18, clean and stable)
===========================================================
Builds directly on v3 which gave 63.6%.
Fixes NaN loss issue from v4.
Target: 66-70%
 
Changes from v3:
  - Mixup augmentation (proper soft-label implementation, NaN-safe)
  - OneCycleLR with correct step order
  - Dropout 0.5 (was 0.4) — closes train/val gap
  - weight_decay 5e-4 (was 1e-4)
  - Frozen BatchNorm during Phase 2
  - Class weights cast to float32 explicitly (NaN fix)
  - autocast disabled during loss computation (NaN fix)
 
Run: python FED_train_v5.py
"""
 
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import GradScaler, autocast
 
if __name__ == '__main__':
 
    # ── 1. GPU ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
 
    # ── 2. Config ─────────────────────────────────────────────────────────────
    TRAIN_DIR = r"C:\Users\HP\OneDrive\Desktop\Projects done\emotion-recognition\emotion_dataset\train"
    TEST_DIR  = r"C:\Users\HP\OneDrive\Desktop\Projects done\emotion-recognition\emotion_dataset\test"
 
    IMG_SIZE        = 64
    BATCH_SIZE      = 64    # ResNet18 fits 64 fine in 4GB
    EPOCHS_FROZEN   = 8
    EPOCHS_FINETUNE = 45
    LR_HEAD         = 3e-3
    LR_FINETUNE     = 1e-4
    NUM_WORKERS     = 0
    MIXUP_ALPHA     = 0.2   # conservative — enough to help, not enough to destabilise
 
    CLASS_NAMES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    NUM_CLASSES = len(CLASS_NAMES)
    OUTPUT_DIR  = "outputs_v5"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    # ── 3. Transforms ─────────────────────────────────────────────────────────
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08))
    ])
 
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
 
    # ── 4. Datasets & loaders ─────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transforms)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
 
    print(f"\nTrain : {len(train_dataset)} | Test : {len(test_dataset)}")
    print(f"Classes: {train_dataset.class_to_idx}")
 
    # ── 5. Class weights — mild, float32 explicit (prevents NaN) ─────────────
    labels = [l for _, l in train_dataset.samples]
    base_w = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    adj_w  = base_w.copy()
    adj_w[CLASS_NAMES.index('fear')] *= 1.3   # mild boost only
    adj_w[CLASS_NAMES.index('sad')]  *= 1.2
    # CRITICAL: cast to float32 — float64 weights cause NaN with autocast
    weights_t = torch.tensor(adj_w, dtype=torch.float32).to(device)
    print(f"\nClass weights: {dict(zip(CLASS_NAMES, adj_w.round(3)))}")
 
    # ── 6. Model: ResNet18 ────────────────────────────────────────────────────
    def build_model():
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad = False
        m.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),      # was 0.4 in v3
            nn.Linear(256, NUM_CLASSES)
        )
        return m
 
    def freeze_bn(model):
        """Keep BatchNorm in eval mode — preserves ImageNet stats during fine-tune."""
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False
 
    model = build_model().to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nResNet18 — Total: {total:,} | Trainable (head): {trainable:,}")
 
    # ── 7. Loss (no autocast on loss — NaN fix) ───────────────────────────────
    # CrossEntropyLoss with float16 intermediate activations can overflow to NaN.
    # Solution: compute logits in float16 (fast), cast back to float32 for loss.
    criterion    = nn.CrossEntropyLoss(weight=weights_t)
    criterion_sm = nn.CrossEntropyLoss(weight=weights_t, label_smoothing=0.1)
    scaler       = GradScaler(device='cuda')
 
    # ── 8. Mixup (NaN-safe, no autocast inside) ───────────────────────────────
    def mixup_forward(model, images, labels, crit):
        """
        Runs one mixup training step.
        Returns loss (float32, no NaN risk) + hard-label accuracy for logging.
        """
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        idx = torch.randperm(images.size(0), device=device)
 
        mixed = lam * images + (1 - lam) * images[idx]
 
        # Forward in float16 via autocast
        with autocast(device_type='cuda'):
            logits = model(mixed)
 
        # Cast to float32 BEFORE computing loss — prevents NaN
        logits_f32 = logits.float()
        labels_a   = labels
        labels_b   = labels[idx]
        loss = lam * crit(logits_f32, labels_a) + (1 - lam) * crit(logits_f32, labels_b)
 
        # Accuracy on original (un-mixed) images for logging
        with torch.no_grad():
            with autocast(device_type='cuda'):
                orig_logits = model(images).float()
        acc = (orig_logits.argmax(1) == labels).float().mean().item()
 
        return loss, acc
 
    # ── 9. Train / eval ───────────────────────────────────────────────────────
    def train_epoch(model, loader, optimizer, scheduler,
                    use_mixup=False, crit=None, onecycle=False):
        model.train()
        if use_mixup:
            freeze_bn(model)
 
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
 
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
 
            if use_mixup:
                loss, acc = mixup_forward(model, images, labels, crit)
            else:
                with autocast(device_type='cuda'):
                    logits = model(images)
                loss = crit(logits.float(), labels)   # float32 for loss
                acc  = (logits.argmax(1) == labels).float().mean().item()
 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
 
            if onecycle:
                scheduler.step()   # OneCycleLR steps every BATCH not epoch
 
            total_loss += loss.item()
            total_acc  += acc
            n_batches  += 1
 
            if (batch_idx + 1) % 100 == 0:
                print(f"  [{batch_idx+1}/{len(loader)}] "
                      f"loss={total_loss/n_batches:.4f} "
                      f"acc={total_acc/n_batches:.4f}", end="\r")
 
        return total_loss / n_batches, total_acc / n_batches
 
    def evaluate(model, loader, crit):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type='cuda'):
                    logits = model(images).float()   # float32 for loss
                total_loss += crit(logits, labels).item() * images.size(0)
                correct    += (logits.argmax(1) == labels).sum().item()
                total      += images.size(0)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return total_loss / total, correct / total, all_preds, all_labels
 
    def run_phase(model, loader, optimizer, scheduler, epochs,
                  name, crit, best_acc=0.0, use_mixup=False, onecycle=False):
        history  = {k: [] for k in ["train_acc","val_acc","train_loss","val_loss"]}
        savepath = os.path.join(OUTPUT_DIR, f"best_{name}.pth")
        patience_ctr, patience = 0, 12
 
        for epoch in range(1, epochs + 1):
            t0 = time.time()
 
            tr_loss, tr_acc = train_epoch(model, loader, optimizer, scheduler,
                                          use_mixup=use_mixup, crit=crit,
                                          onecycle=onecycle)
            vl_loss, vl_acc, _, _ = evaluate(model, test_loader, crit)
 
            # Epoch-level step only for ReduceLROnPlateau
            if not onecycle:
                scheduler.step(vl_loss)
 
            for k, v in zip(["train_acc","val_acc","train_loss","val_loss"],
                            [tr_acc, vl_acc, tr_loss, vl_loss]):
                history[k].append(v)
 
            marker = ""
            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(model.state_dict(), savepath)
                marker = " ← best"
                patience_ctr = 0
            else:
                patience_ctr += 1
 
            lr = optimizer.param_groups[0]['lr']
            print(f"[{name}] Ep {epoch:02d}/{epochs} | "
                  f"train={tr_acc:.4f} val={vl_acc:.4f} | "
                  f"loss={tr_loss:.4f}/{vl_loss:.4f} | "
                  f"lr={lr:.2e} | {int(time.time()-t0)}s{marker}")
 
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
 
        model.load_state_dict(torch.load(savepath, weights_only=True))
        print(f"\n{name} best val: {best_acc:.4f} ({best_acc*100:.2f}%)")
        return history, best_acc
 
    # ── 10. Phase 1: Head only ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("PHASE 1 — Head only | frozen base | clean signal")
    print("="*65)
 
    opt1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=5e-4
    )
    sch1 = optim.lr_scheduler.ReduceLROnPlateau(
        opt1, mode='min', factor=0.5, patience=2
    )
    h1, best_acc = run_phase(model, train_loader, opt1, sch1,
                              EPOCHS_FROZEN, "phase1", criterion,
                              use_mixup=False, onecycle=False)
 
    # ── 11. Phase 2: Full fine-tune + Mixup + OneCycleLR ─────────────────────
    print("\n" + "="*65)
    print("PHASE 2 — Full fine-tune | Mixup | OneCycleLR | frozen BN")
    print("="*65)
 
    # Unfreeze all, then re-freeze BN
    for p in model.parameters():
        p.requires_grad = True
    freeze_bn(model)
 
    bn_count  = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} | BN layers frozen: {bn_count}")
 
    backbone_p = [p for n, p in model.named_parameters()
                  if 'fc' not in n and p.requires_grad]
    head_p     = [p for n, p in model.named_parameters()
                  if 'fc' in n]
 
    opt2 = optim.AdamW([
        {'params': backbone_p, 'lr': LR_FINETUNE},
        {'params': head_p,     'lr': LR_FINETUNE * 5},
    ], weight_decay=5e-4)
 
    # OneCycleLR — steps every batch, not every epoch
    total_steps = EPOCHS_FINETUNE * len(train_loader)
    sch2 = optim.lr_scheduler.OneCycleLR(
        opt2,
        max_lr=[LR_FINETUNE, LR_FINETUNE * 5],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=1000
    )
 
    h2, best_acc = run_phase(model, train_loader, opt2, sch2,
                              EPOCHS_FINETUNE, "phase2", criterion_sm,
                              best_acc=best_acc,
                              use_mixup=True, onecycle=True)
 
    # ── 12. Training curves ───────────────────────────────────────────────────
    ta = h1["train_acc"]  + h2["train_acc"]
    va = h1["val_acc"]    + h2["val_acc"]
    tl = h1["train_loss"] + h2["train_loss"]
    vl = h1["val_loss"]   + h2["val_loss"]
    sp = len(h1["train_acc"])
    x  = range(1, len(ta) + 1)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(x, ta, label="Train",      lw=2, color='steelblue')
    ax1.plot(x, va, label="Validation", lw=2, color='darkorange')
    ax1.axvline(x=sp, color='gray', ls='--', alpha=0.6, label='Fine-tune start')
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
    ax1.set_ylim([0, 1]); ax1.legend(); ax1.grid(alpha=0.3)
 
    ax2.plot(x, tl, label="Train",      lw=2, color='steelblue')
    ax2.plot(x, vl, label="Validation", lw=2, color='darkorange')
    ax2.axvline(x=sp, color='gray', ls='--', alpha=0.6, label='Fine-tune start')
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(alpha=0.3)
 
    plt.suptitle("ResNet18 v5 — Mixup + OneCycleLR + frozen BN", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()
 
    # ── 13. Final evaluation ──────────────────────────────────────────────────
    print("\nFinal evaluation...")
    _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion_sm)
 
    print(f"\n{'='*55}")
    print(f"v5 Final Accuracy   : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"v3 (ResNet18 clean) : 0.6362 (63.62%)")
    print(f"FER2013 SOTA        : 0.7370 (73.70%)")
    print(f"Baseline (random)   : {1/6:.4f} ({100/6:.2f}%)")
    print(f"{'='*55}\n")
 
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
 
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm,      annot=True, fmt="d",   cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")
 
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalised)")
    axes[1].set_ylabel("True"); axes[1].set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()
 
    per_class = cm_norm.diagonal()
    print(f"\nPer-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        bar  = "█" * int(per_class[i] * 30)
        flag = " ← needs work" if per_class[i] < 0.5 else ""
        print(f"  {name:<10}: {bar:<30} {per_class[i]*100:.1f}%{flag}")
    print(f"\nBest : {CLASS_NAMES[np.argmax(per_class)]} ({per_class.max()*100:.1f}%)")
    print(f"Worst: {CLASS_NAMES[np.argmin(per_class)]} ({per_class.min()*100:.1f}%)")
 
    # ── 14. Save ──────────────────────────────────────────────────────────────
    torch.save(model, os.path.join(OUTPUT_DIR, "emotion_model_v5.pth"))
    torch.save(model.state_dict(),
               os.path.join(OUTPUT_DIR, "emotion_model_v5_weights.pth"))
    print(f"\nSaved → outputs_v5/emotion_model_v5.pth")
    print("Done!")
 