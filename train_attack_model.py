#!/usr/bin/env python
import random
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18

class MembershipDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.transform = kwargs.get("transform", None)

    def __len__(self):
        for name in ["ids", "sample_ids", "data", "images", "imgs", "x"]:
            if hasattr(self, name):
                values = getattr(self, name)
                if values is not None:
                    return len(values)

        raise RuntimeError("Could not determine dataset length.")

    def __getitem__(self, index):
        sample_id = self._safe_get(
            ["ids", "sample_ids", "id"],
            index,
            default=index,
        )

        image = self._safe_get(
            ["data", "images", "imgs", "x"],
            index,
            default=None,
        )

        label = self._safe_get(
            ["labels", "targets", "y"],
            index,
            default=0,
        )

        membership = self._safe_get(
            ["memberships", "membership", "member", "is_member"],
            index,
            default=0,
        )

        if sample_id is None:
            sample_id = index

        if image is None:
            raise RuntimeError(f"Image is None at dataset index {index}")

        if label is None:
            label = 0

        if membership is None:
            membership = 0

        image = self._prepare_image(image)

        if self.transform is not None:
            image = self.transform(image)

        return int(sample_id), image, int(label), int(membership)

    def _safe_get(self, possible_names, index, default=None):
        for name in possible_names:
            if not hasattr(self, name):
                continue

            values = getattr(self, name)

            if values is None:
                continue

            try:
                value = values[index]
            except Exception:
                continue

            if value is None:
                continue

            return value

        return default

    def _prepare_image(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported image type: {type(image)}")

        image = image.float()

        if image.ndim == 2:
            image = image.unsqueeze(0)

        if image.ndim == 3:
            if image.shape[0] not in [1, 3] and image.shape[-1] in [1, 3]:
                image = image.permute(2, 0, 1)

        if image.ndim != 3:
            raise RuntimeError(f"Expected image shape [C,H,W], got {tuple(image.shape)}")

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.max() > 1.0:
            image = image / 255.0

        return image

API_KEY = "613bfba4b9ba5a4c6bce1eeecb39f227"
TASK_ID = "01-mia"
LEADERBOARD_URL = "http://35.192.205.84:80/submit"

NUM_SHADOW_MODELS = 64
SHADOW_EPOCHS = 50
SHADOW_BATCH_SIZE = 128
SHADOW_LR = 0.03
SHADOW_SUBSET_FRAC = 0.5
EVAL_BATCH_SIZE = 64
BASE = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}", flush=True)


MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize((32, 32), antialias=True),
    transforms.Normalize(mean=MEAN, std=STD),
])

print("Loading datasets...", flush=True)

pub_ds = torch.load(BASE / "pub.pt", weights_only=False)
priv_ds = torch.load(BASE / "priv.pt", weights_only=False)

pub_ds.transform = transform
priv_ds.transform = transform

pub_loader = DataLoader(
    pub_ds,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

priv_loader = DataLoader(
    priv_ds,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


def build_model(num_classes=9):
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)

    return model


print("Loading original model...", flush=True)

orig_model = build_model()
orig_model.load_state_dict(torch.load(BASE / "model.pt", map_location="cpu"))
orig_model.to(DEVICE)
orig_model.eval()



def get_losses(model, loader):
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction="none")

    id_to_loss = {}

    with torch.no_grad():
        for batch in loader:
            ids, images, labels, *_ = batch

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            losses = loss_function(logits, labels)

            for sample_id, loss_value in zip(ids.tolist(), losses.cpu().tolist()):
                id_to_loss[int(sample_id)] = float(loss_value)

    return id_to_loss


def get_confidence_scores(loader):
    scores = {}

    orig_model.eval()

    with torch.no_grad():
        for batch in loader:
            ids, images, *_ = batch

            images = images.to(DEVICE)

            logits = orig_model(images)
            probabilities = F.softmax(logits, dim=1)
            confidence_values = probabilities.max(dim=1).values.cpu().tolist()

            for sample_id, confidence in zip(ids.tolist(), confidence_values):
                scores[int(sample_id)] = float(confidence)

    return scores


def normalise_scores(score_dict):
    values = np.array(list(score_dict.values()), dtype=np.float64)

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return {sample_id: 0.5 for sample_id in score_dict}

    return {
        sample_id: float((score - min_value) / (max_value - min_value))
        for sample_id, score in score_dict.items()
    }


def get_index_to_id(dataset):
    index_to_id = {}

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    current_index = 0

    for batch in loader:
        ids = batch[0].tolist()

        for sample_id in ids:
            index_to_id[current_index] = int(sample_id)
            current_index += 1

    return index_to_id


def train_shadow_model(train_indices, num_classes=9):
    shadow_dataset = Subset(pub_ds, train_indices)

    loader = DataLoader(
        shadow_dataset,
        batch_size=SHADOW_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes).to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=SHADOW_LR,
        momentum=0.9,
        weight_decay=5e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SHADOW_EPOCHS,
    )

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(SHADOW_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in loader:
            _, images, labels, *_ = batch

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        scheduler.step()

        average_loss = total_loss / max(len(loader), 1)

        print(
            f"    Epoch {epoch + 1}/{SHADOW_EPOCHS} | loss={average_loss:.4f}",
            flush=True,
        )

    return model


def run_lira():
    number_of_public_samples = len(pub_ds)
    shadow_train_size = int(number_of_public_samples * SHADOW_SUBSET_FRAC)

    all_public_indices = list(range(number_of_public_samples))
    index_to_public_id = get_index_to_id(pub_ds)

    public_in_losses = {}
    public_out_losses = {}
    private_losses = {}

    all_public_ids = set()

    for shadow_index in range(NUM_SHADOW_MODELS):
        print(
            f"Training shadow model {shadow_index + 1}/{NUM_SHADOW_MODELS}...",
            flush=True,
        )

        shuffled_indices = all_public_indices.copy()
        random.shuffle(shuffled_indices)

        in_indices = shuffled_indices[:shadow_train_size]
        out_indices = shuffled_indices[shadow_train_size:]

        shadow_model = train_shadow_model(in_indices)

        print("  Evaluating shadow model on public data...", flush=True)
        shadow_public_losses = get_losses(shadow_model, pub_loader)

        for sample_id in shadow_public_losses:
            all_public_ids.add(int(sample_id))

        for index in in_indices:
            sample_id = index_to_public_id[index]

            if sample_id in shadow_public_losses:
                public_in_losses.setdefault(sample_id, []).append(
                    shadow_public_losses[sample_id]
                )

        for index in out_indices:
            sample_id = index_to_public_id[index]

            if sample_id in shadow_public_losses:
                public_out_losses.setdefault(sample_id, []).append(
                    shadow_public_losses[sample_id]
                )

        print("  Evaluating shadow model on private data...", flush=True)
        shadow_private_losses = get_losses(shadow_model, priv_loader)

        for sample_id, loss_value in shadow_private_losses.items():
            private_losses.setdefault(sample_id, []).append(loss_value)

        del shadow_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    public_scores = {}

    for sample_id in all_public_ids:
        in_losses = public_in_losses.get(sample_id, [])
        out_losses = public_out_losses.get(sample_id, [])

        if len(in_losses) > 0 and len(out_losses) > 0:
            score = np.mean(out_losses) - np.mean(in_losses)
        elif len(in_losses) > 0:
            score = -np.mean(in_losses)
        elif len(out_losses) > 0:
            score = np.mean(out_losses)
        else:
            score = 0.0

        public_scores[sample_id] = float(score)

    private_scores = {}

    for sample_id, losses in private_losses.items():
        average_loss = float(np.mean(losses))

        # Lower loss means the sample looks more member-like.
        private_scores[sample_id] = -average_loss

    return public_scores, private_scores


def evaluate_public_scores(scores):
    member_scores = []
    non_member_scores = []

    loader = DataLoader(
        pub_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    for batch in loader:
        ids, _, _, memberships = batch

        for sample_id, membership in zip(ids.tolist(), memberships.tolist()):
            sample_id = int(sample_id)

            if sample_id not in scores:
                continue

            if int(membership) == 1:
                member_scores.append(scores[sample_id])
            else:
                non_member_scores.append(scores[sample_id])

    member_scores = np.array(member_scores)
    non_member_scores = np.array(non_member_scores)

    if len(member_scores) == 0 or len(non_member_scores) == 0:
        print("Could not evaluate public scores. Missing member/non-member labels.")
        return

    all_scores = np.concatenate([member_scores, non_member_scores])
    thresholds = np.sort(all_scores)[::-1]

    best_tpr = 0.0
    target_fpr = 0.05

    for threshold in thresholds:
        tpr = float((member_scores >= threshold).mean())
        fpr = float((non_member_scores >= threshold).mean())

        if fpr <= target_fpr:
            best_tpr = max(best_tpr, tpr)

    print(
        f"\n[Validation] TPR @ 5% FPR on pub.pt = {best_tpr:.4f}",
        flush=True,
    )

def submit_scores(scores):
    rows = sorted(scores.items())

    csv_lines = ["id,score"]

    for sample_id, score in rows:
        csv_lines.append(f"{sample_id},{score:.6f}")

    csv_content = "\n".join(csv_lines)

    output_path = BASE / "submission.csv"

    with open(output_path, "w", newline="") as file:
        file.write(csv_content)

    print(f"Saved submission.csv to: {output_path}", flush=True)

    response = requests.post(
        LEADERBOARD_URL,
        headers={"x-api-key": API_KEY},
        data={"task_id": TASK_ID},
        files={"file": ("submission.csv", csv_content, "text/csv")},
    )

    print(
        f"\nSubmission response ({response.status_code}): {response.text}",
        flush=True,
    )

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 60)
    print("LiRA Membership Inference Attack - TML 2026")
    print("=" * 60)

    print("\n[1/4] Computing original model confidence scores...", flush=True)
    public_confidence = get_confidence_scores(pub_loader)
    private_confidence = get_confidence_scores(priv_loader)

    print("\n[2/4] Running LiRA...", flush=True)
    public_lira, private_lira = run_lira()

    print("\n[3/4] Combining LiRA and confidence scores...", flush=True)

    public_confidence_normalised = normalise_scores(public_confidence)
    private_confidence_normalised = normalise_scores(private_confidence)

    public_lira_normalised = normalise_scores(public_lira)
    private_lira_normalised = normalise_scores(private_lira)

    public_final_scores = {}

    for sample_id in public_lira_normalised:
        public_final_scores[sample_id] = (
            0.5 * public_lira_normalised[sample_id]
            + 0.5 * public_confidence_normalised.get(sample_id, 0.5)
        )

    private_final_scores = {}

    for sample_id in private_lira_normalised:
        private_final_scores[sample_id] = (
            0.5 * private_lira_normalised[sample_id]
            + 0.5 * private_confidence_normalised.get(sample_id, 0.5)
        )

    print("\n[4/4] Evaluating public attack quality...", flush=True)
    evaluate_public_scores(public_final_scores)

    print("\nSubmitting private scores...", flush=True)
    submit_scores(private_final_scores)

    print("\nDone.", flush=True)
