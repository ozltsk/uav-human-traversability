"""Нейромережева оцінка прохідності завалів для людини.

Повнозгорткова мережа типу U-Net отримує сім шарів карти (висота опори,
просвіт, частки невідомих вокселів у смугах трьох поз, щільність зайнятих
вокселів, частка оглянутих вокселів) і для кожної клітинки передбачає
розподіл імовірностей п'яти станів: прохідна пішки, лише пригнувшись,
лише повзком, непрохідна, недостатньо оглянута. Навчальні приклади дає
генератор синтетичних сцен, де точний стан кожної клітинки відомий.
Рішення про стан приймається обережно: поза вважається допустимою,
лише якщо ймовірність того, що клітинка прохідна принаймні в цій позі,
не менша за поріг theta.
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pose_model import WALK, CROUCH, CRAWL, BLOCKED, UNSEEN

N_CLASSES = 5
N_FEATURES = 7


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TravNet(nn.Module):
    """U-Net з трьома рівнями: 16-32-64 каналів, вихід -- логіти 5 класів."""

    def __init__(self, n_in=N_FEATURES, n_out=N_CLASSES, width=16):
        super().__init__()
        w = width
        self.enc1 = _block(n_in, w)
        self.enc2 = _block(w, 2 * w)
        self.enc3 = _block(2 * w, 4 * w)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = _block(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = _block(2 * w, w)
        self.head = nn.Conv2d(w, n_out, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, path, augment=False):
        z = np.load(path)
        self.X = z["X"].astype(np.float32)
        self.Y = z["Y"].astype(np.int64)
        self.augment = augment

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        if self.augment:                      # віддзеркалення вздовж y
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1].copy(), y[:, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def class_weights(Y, boost_blocked=2.0):
    """Ваги класів: обернені до частот, непрохідний клас додатково підсилено,
    бо хибно прохідна клітинка небезпечніша за пропущену."""
    counts = np.bincount(Y.reshape(-1), minlength=N_CLASSES).astype(np.float64) + 1.0
    w = counts.sum() / (N_CLASSES * counts)
    w[BLOCKED] *= boost_blocked
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train(train_path, val_path, out_path, epochs=12, batch=8, lr=1e-3, width=16, device="cpu", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds_tr = SceneDataset(train_path, augment=True)
    ds_va = SceneDataset(val_path)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=batch)
    model = TravNet(width=width).to(device)
    w = class_weights(ds_tr.Y).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    best = (1e9, None)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TravNet: {n_params} параметрів, навчальних сцен {len(ds_tr)}, валідаційних {len(ds_va)}")
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tr_loss, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=w)
            loss.backward()
            opt.step()
            tr_loss += float(loss) * x.shape[0]
            n += x.shape[0]
        sched.step()
        va_loss, va_acc, va_false_pass = evaluate_loader(model, dl_va, w, device)
        history.append({"epoch": ep, "train_loss": tr_loss / n, "val_loss": va_loss, "val_acc": va_acc,
                        "val_false_passable": va_false_pass, "time_s": time.perf_counter() - t0})
        print(f"епоха {ep:2d}: втрати навч. {tr_loss / n:.4f}, вал. {va_loss:.4f}, точність вал. {va_acc:.4f}, "
              f"хибно прохідних {va_false_pass:.4f}, {time.perf_counter() - t0:.1f} с")
        if va_loss < best[0]:
            best = (va_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    torch.save({"state_dict": model.state_dict(), "width": width, "history": history}, out_path)
    with open(out_path + ".history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)
    return model, history


def evaluate_loader(model, loader, w, device):
    model.eval()
    loss, n, correct, total, false_pass, blocked = 0.0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss += float(F.cross_entropy(logits, y, weight=w)) * x.shape[0]
            n += x.shape[0]
            pred = logits.argmax(1)
            correct += int((pred == y).sum())
            total += y.numel()
            is_blocked = y == BLOCKED
            blocked += int(is_blocked.sum())
            false_pass += int(((pred <= CRAWL) & is_blocked).sum())
    return loss / n, correct / total, false_pass / max(1, blocked)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = TravNet(width=ck.get("width", 16)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def predict_probs(model, features, device="cpu"):
    """Імовірності станів (5, nx, ny) для тензора ознак (7, nx, ny)."""
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))[None].to(device)
    nx, ny = x.shape[2], x.shape[3]
    px, py = (-nx) % 4, (-ny) % 4            # доповнення до кратності 4 для U-Net
    x = F.pad(x, (0, py, 0, px), mode="replicate")
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0, :, :nx, :ny].cpu().numpy()
    return p


def decide(probs, theta=0.6, unknown_crawl=None, u_unseen=0.9, voxel=0.1, mu=0.0, check_width=True):
    """Обережне рішення: поза k допустима, якщо P(стан <= k) >= theta, а
    відстань до найближчої недопустимої для цієї пози клітинки перевищує
    половину ширини тіла (перетворення відстані). Клітинка, для якої жодна
    поза не допустима, вважається непрохідною, а за великої частки невідомих
    вокселів -- недостатньо оглянутою."""
    from pose_model import width_check
    cum = np.cumsum(probs[:CRAWL + 1], axis=0)
    state = np.full(probs.shape[1:], BLOCKED, dtype=np.int8)
    for k in (CRAWL, CROUCH, WALK):
        adm = cum[k] >= theta
        if check_width:
            adm = width_check(adm, k, voxel, mu)
        state[adm] = k
    unseen = (state == BLOCKED) & (probs[UNSEEN] > probs[BLOCKED])
    if unknown_crawl is not None:
        unseen |= (state == BLOCKED) & (unknown_crawl > u_unseen)
    state[unseen] = UNSEEN
    return state
