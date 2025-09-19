
import math
import numpy as np
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))


    def forward(self, x):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var  = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        x_hat = x_hat * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return x_hat

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = CustomBatchNorm(out_ch)
        self.act1  = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = CustomBatchNorm(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.out_act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.skip(identity)
        out = self.out_act(out + identity)
        return out


class MyResNet(nn.Module):
    def __init__(self, num_classes=10, base_channels=32):
        super().__init__()
        # Block 1: [B,3,32,32] → [B,F,32,32]
        self.block1 = ResidualBlock(3, base_channels)
        self.pool1  = nn.AvgPool2d(2)   # → [B,F,16,16]

        # Block 2
        self.block2 = ResidualBlock(base_channels, base_channels*2)
        self.pool2  = nn.AvgPool2d(2)   # → [B,2F,8,8]

        # Block 3
        self.block3 = ResidualBlock(base_channels*2, base_channels*4)

        # Classifier
        self.fc1 = nn.Linear(base_channels*4*8*8, 256)
        self.act_fc1 = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = torch.flatten(x, 1)
        x = self.act_fc1(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


@torch.no_grad()
def save_first_50_test_images(model, X_test, y_test, test_acc_final, BATCH_SIZE, CLASS_NAMES, OUTDIR):
    model.eval()
    # first 50 samples
    probs_all = model(X_test[:50])
    preds_all = probs_all.argmax(dim=1).cpu().numpy()
    labels_all = y_test[:50].cpu().numpy()


    acc_text = f"{test_acc_final:.2f}"

    # ---- save first 50 images ----
    for i in range(50):
        img_chw = X_test[i].cpu().numpy()
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        plt.figure(figsize=(3, 3))
        plt.imshow(img_hwc)
        plt.axis("off")
        label_name = CLASS_NAMES[labels_all[i]]
        pred_name  = CLASS_NAMES[preds_all[i]]
        title = f"test {i:05d}  label {label_name}  pred {pred_name}  accuracy {acc_text}"
        plt.title(title, fontsize=8)
        plt.tight_layout(pad=0.1)
        fname = os.path.join(OUTDIR, f"test_{i:05d}.png")
        plt.savefig(fname, dpi=150)
        plt.close()



def main():

    BATCH_SIZE = 100
    NUM_EPOCHS = 30
    INIT_LR = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    MILESTONES = [15, 25]  # drop LR at these epochs
    GAMMA = 0.1
    OUTDIR = "output"
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = "./data"
    cifar_dir = os.path.join(data_root, "cifar-10-batches-py")

    # Only download if not already present
    download_flag = not os.path.exists(cifar_dir)

    train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=download_flag, transform=None)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=download_flag, transform=None)

    if hasattr(train_full, "data") and hasattr(train_full, "targets"):
        # Fast path
        X_train_full = torch.as_tensor(train_full.data).permute(0, 3, 1, 2).float().div(255.0)
        y_train_full = torch.as_tensor(train_full.targets, dtype=torch.long)

        X_test = torch.as_tensor(test_set.data).permute(0, 3, 1, 2).float().div(255.0)
        y_test = torch.as_tensor(test_set.targets, dtype=torch.long)
    else:
        # Fallback
        tr = torchvision.transforms.ToTensor()
        train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=download_flag, transform=tr)
        test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=download_flag, transform=tr)

        X_train_full = torch.stack([train_full[i][0] for i in range(len(train_full))])
        y_train_full = torch.tensor([train_full[i][1] for i in range(len(train_full))], dtype=torch.long)

        X_test = torch.stack([test_set[i][0] for i in range(len(test_set))])
        y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))], dtype=torch.long)

    X_train, y_train = X_train_full[:45000], y_train_full[:45000]
    X_val, y_val = X_train_full[45000:], y_train_full[45000:]

    CLASS_NAMES = train_full.classes

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    model = MyResNet(num_classes=10, base_channels=32).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=INIT_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=GAMMA
    )

    train_acc_hist, val_acc_hist, test_acc_hist = [], [], []
    train_loss_hist, val_loss_hist = [], []

    num_batches = X_train.shape[0] // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss_sum, seen_in_epoch = 0.0, 0
        train_correct = 0

        for batch in range(num_batches):
            optimizer.zero_grad()
            s, e = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE
            xb, yb = X_train[s:e], y_train[s:e]

            probs = model(xb)  # shape [B, C], already softmaxed

            eps = 1e-9
            one_hot = F.one_hot(yb, num_classes=10).float()
            log_probs = torch.log(probs.clamp_min(eps))
            loss = -(one_hot * log_probs).sum(dim=1).mean()


            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * xb.shape[0]
            seen_in_epoch += xb.shape[0]
            train_correct += (probs.argmax(dim=1) == yb).sum().item()

        scheduler.step()

        tr_loss_avg = epoch_loss_sum / seen_in_epoch
        tr_acc_eval = 100.0 * train_correct / seen_in_epoch

        # validation
        model.eval()
        val_loss_sum, val_seen, val_correct = 0.0, 0, 0
        with torch.no_grad():
            n_val = X_val.size(0)
            nb_val = math.ceil(n_val / BATCH_SIZE)
            for b in range(nb_val):
                s = b * BATCH_SIZE
                e = s + BATCH_SIZE
                xb, yb = X_val[s:e], y_val[s:e]

                probs = model(xb)  # [B, C]

                eps = 1e-9
                one_hot = F.one_hot(yb, num_classes=10).float()
                log_probs = torch.log(probs.clamp_min(eps))
                vloss = -(one_hot * log_probs).sum(dim=1).mean()

                val_loss_sum += vloss.item() * xb.size(0)
                val_seen += xb.size(0)
                val_correct += (probs.argmax(dim=1) == yb).sum().item()

        va_loss_eval = val_loss_sum / val_seen
        va_acc_eval = 100.0 * val_correct / val_seen

        train_loss_hist.append(tr_loss_avg)
        val_loss_hist.append(va_loss_eval)
        train_acc_hist.append(tr_acc_eval)
        val_acc_hist.append(va_acc_eval)

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss {tr_loss_avg:.4f} Acc {tr_acc_eval:.2f}% | "
              f"Val Loss {va_loss_eval:.4f} Acc {va_acc_eval:.2f}% " )

    epochs = np.arange(1, NUM_EPOCHS + 1)

    # test loss & accuracy
    model.eval()
    with torch.no_grad():
        n_te = X_test.size(0)
        nb_te = n_te // BATCH_SIZE
        test_loss_sum, test_seen, test_correct = 0.0, 0, 0

        for b in range(nb_te):
            s = b * BATCH_SIZE
            e = s + BATCH_SIZE
            probs = model(X_test[s:e])

            eps = 1e-9
            one_hot = F.one_hot(y_test[s:e], num_classes=10).float()
            log_probs = torch.log(probs.clamp_min(eps))
            vloss = -(one_hot * log_probs).sum(dim=1).mean()

            test_loss_sum += vloss.item() * (e - s)
            test_seen += (e - s)
            test_correct += (probs.argmax(dim=1) == y_test[s:e]).sum().item()

        test_loss_final = test_loss_sum / test_seen
        test_acc_final = 100.0 * test_correct / test_seen


    plt.figure()
    plt.plot(epochs, train_acc_hist, label="train")
    plt.plot(epochs, val_acc_hist, label="valid")
    plt.axhline(test_acc_final, linestyle="--", linewidth=1.5,
                label=f"test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("ResNet-3 Accuracy")
    plt.savefig(os.path.join(OUTDIR, "accuracy.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_loss_hist, label="train")
    plt.plot(epochs, val_loss_hist, label="validation")
    plt.axhline(test_loss_final, linestyle="--", linewidth=1.5,
                label=f"test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("ResNet-3 Loss")
    plt.savefig(os.path.join(OUTDIR, "loss.png"))
    plt.close()

    save_first_50_test_images(model, X_test, y_test, test_acc_final, BATCH_SIZE, CLASS_NAMES, OUTDIR)
    print(f"Done. Files saved in '{OUTDIR}/'")

if __name__ == "__main__":
    main()
