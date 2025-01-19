import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torch import matrix_exp
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1) DEFINE A DISCRETE GROUP OF IMAGE TRANSFORMS
GENERATOR_SYMBOLS = {
    0: "R",
    1: "Ri",
    2: "F",
    3: "Fi",
    4: "T",
    5: "Ti",
}
NUM_GENERATORS = len(GENERATOR_SYMBOLS)

def apply_transform(img, word):
    for g in word:
        if g == 0:
            img = T.functional.rotate(img, -90)
        elif g == 1:
            img = T.functional.rotate(img, 90)
        elif g == 2 or g == 3:
            img = T.functional.hflip(img)
        elif g == 4:
            img = T.functional.pad(img, (2, 0, 0, 0), fill=0)
            img = T.functional.crop(img, top=0, left=0, height=32, width=32)
        elif g == 5:
            img = T.functional.pad(img, (0, 0, 2, 0), fill=0)
            img = T.functional.crop(img, top=0, left=0, height=32, width=32)
    return img

# 2) DATASET CLASS APPLYING RANDOM GROUP TRANSFORMS
class CIFARTransformed(Dataset):
    def __init__(self, root, train=True, download=True, max_word_len=3):
        self.data = CIFAR10(root=root, train=train, download=download)
        self.transform = T.ToTensor()
        self.max_word_len = max_word_len
        self.num_gens = NUM_GENERATORS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        L = random.randint(0, self.max_word_len)
        word = [random.randint(0, self.num_gens - 1) for _ in range(L)]
        img = apply_transform(img, word)
        x_img = self.transform(img)
        enc = torch.zeros(self.max_word_len, self.num_gens)
        for i, g_idx in enumerate(word):
            s = 1.0 if g_idx % 2 == 0 else -1.0
            enc[i, g_idx] = s
        return enc, x_img, label

# 3) MATRIXNET MODULE
class MatrixNet(nn.Module):
    def __init__(self, num_generators, hidden_dim=64, matrix_size=4, out_dim=32):
        super().__init__()
        self.l1 = nn.Linear(num_generators, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, matrix_size * matrix_size, bias=False)
        self.matrix_size = matrix_size
        self.out_dim = out_dim
        self.relu = nn.ReLU()
        self.proj = nn.Linear(matrix_size * matrix_size, out_dim)

    def matrix_block(self, x):
        h = self.l1(x)
        h = self.relu(h)
        h = self.l2(h)
        h = h.view(-1, self.matrix_size, self.matrix_size)
        return matrix_exp(h)

    def forward(self, word_enc):
        B, L, G = word_enc.shape
        mat_rep = None
        for i in range(L):
            x_i = word_enc[:, i, :]
            if (x_i.abs().sum(dim=1) == 0).all():
                continue
            Mi = self.matrix_block(x_i)
            if mat_rep is None:
                mat_rep = Mi
            else:
                mat_rep = torch.bmm(mat_rep, Mi)
        if mat_rep is None:
            mat_rep = torch.eye(self.matrix_size, device=word_enc.device)
            mat_rep = mat_rep.unsqueeze(0).repeat(B, 1, 1)
        mat_rep_flat = mat_rep.view(B, -1)
        return self.relu(self.proj(mat_rep_flat))

# 4) CNN + MATRIXNET
class CNNwithMatrixNet(nn.Module):
    def __init__(self, num_gens, mat_hidden=64, mat_size=4, mat_out=32, cnn_out=128, num_classes=10):
        super().__init__()
        self.matrix_net = MatrixNet(num_gens, mat_hidden, mat_size, mat_out)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.linear_cnn = nn.Linear(64 * 4 * 4, cnn_out)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(cnn_out + mat_out, num_classes)

    def forward(self, word_enc, x_img):
        mat_feat = self.matrix_net(word_enc)
        cnn_feat = self.conv(x_img)
        cnn_feat = cnn_feat.view(x_img.size(0), -1)
        cnn_feat = self.relu(self.linear_cnn(cnn_feat))
        fused = torch.cat([cnn_feat, mat_feat], dim=1)
        return self.classifier(fused)

# 5) TRAIN + EVAL FUNCTIONS WITH TQDM
def train_epoch(model, loader, optimizer, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    for enc, x_img, labels in tqdm(loader, desc="Training"):
        enc, x_img, labels = enc.to(device), x_img.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(enc, x_img)
        loss = crit(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * enc.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += enc.size(0)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    avg_loss = total_loss / total
    avg_acc = correct / total
    f1 = f1_score(labels_all, preds_all, average='macro')
    return avg_loss, avg_acc, f1

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    for enc, x_img, labels in tqdm(loader, desc="Evaluating"):
        enc, x_img, labels = enc.to(device), x_img.to(device), labels.to(device)
        logits = model(enc, x_img)
        loss = crit(logits, labels)
        total_loss += loss.item() * enc.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += enc.size(0)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    avg_loss = total_loss / total
    avg_acc = correct / total
    f1 = f1_score(labels_all, preds_all, average='macro')
    return avg_loss, avg_acc, f1, preds_all, labels_all

# 6) EXPERIMENT ROUTINE
def run_experiment(lr, mat_size, mat_hidden, mat_out, epochs, device, train_loader, test_loader):
    model = CNNwithMatrixNet(
        num_gens=NUM_GENERATORS,
        mat_hidden=mat_hidden,
        mat_size=mat_size,
        mat_out=mat_out,
        cnn_out=128,
        num_classes=10
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'test_loss': [],
        'test_acc': [],
        'test_f1': []
    }
    for epoch in range(1, epochs+1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc, test_f1, preds, labels = eval_epoch(model, test_loader, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)
    cm = confusion_matrix(labels, preds)
    return model, history, cm

# 7) PLOTTING
def plot_curves(history, title_suffix=''):
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.legend()
    plt.title(f"MatrixNet Loss vs Epoch {title_suffix}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"MatrixNet_Loss{title_suffix}.png")
    plt.close()

    plt.figure()
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.legend()
    plt.title(f"MatrixNet Accuracy vs Epoch {title_suffix}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f"MatrixNet_Accuracy{title_suffix}.png")
    plt.close()

    plt.figure()
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['test_f1'], label='Test F1')
    plt.legend()
    plt.title(f"MatrixNet F1 vs Epoch {title_suffix}")
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.savefig(f"MatrixNet_F1{title_suffix}.png")
    plt.close()

def plot_confusion_matrix(cmat, title_suffix=''):
    disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, values_format='d', xticks_rotation='vertical')
    plt.title(f"MatrixNet Confusion Matrix {title_suffix}")
    plt.savefig(f"MatrixNet_CM{title_suffix}.png")
    plt.close()

# 8) MAIN SCRIPT
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = CIFARTransformed(root='./data_cifar', train=True, download=True)
    test_set = CIFARTransformed(root='./data_cifar', train=False, download=True)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    param_grid = [
        {'lr':1e-3, 'mat_size':4, 'mat_hidden':64, 'mat_out':32},
        {'lr':3e-4, 'mat_size':4, 'mat_hidden':128, 'mat_out':32},
    ]
    best_f1 = 0
    best_cfg = None
    for cfg in param_grid:
        model, hist, cmat = run_experiment(
            lr=cfg['lr'],
            mat_size=cfg['mat_size'],
            mat_hidden=cfg['mat_hidden'],
            mat_out=cfg['mat_out'],
            epochs=15,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader
        )
        final_f1 = hist['test_f1'][-1]
        if final_f1 > best_f1:
            best_f1 = final_f1
            best_cfg = (cfg, model, hist, cmat)
    cfg, model, history, cmat = best_cfg
    suffix = f"_lr{cfg['lr']}_msize{cfg['mat_size']}"
    plot_curves(history, title_suffix=suffix)
    plot_confusion_matrix(cmat, title_suffix=suffix)
    print("Done. Best Config:", cfg)
    print("Saved plots and confusion matrix.")

if __name__ == "__main__":
    main()
