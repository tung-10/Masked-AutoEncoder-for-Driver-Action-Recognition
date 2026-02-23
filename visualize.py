import matplotlib.pyplot as plt
import json

with open('./checkpoints/log.txt', 'r') as f:
    log_content = f.read()

epochs = []
train_loss = []
val_loss = []
val_acc1 = []
val_acc5 = []

for line in log_content.strip().split('\n'):
    data = json.loads(line)
    epochs.append(data['epoch'])
    train_loss.append(data['train_loss'])
    val_loss.append(data['val_loss'])
    val_acc1.append(data['val_acc1'])
    val_acc5.append(data['val_acc5'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(epochs, train_loss, label='Train Loss', marker='o', color='tab:blue')
ax1.plot(epochs, val_loss, label='Val Loss', marker='o', color='tab:orange')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training vs Validation Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(epochs, val_acc1, label='Val Acc Top-1', marker='s', color='tab:green')
ax2.plot(epochs, val_acc5, label='Val Acc Top-5', marker='^', color='tab:red')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Validation Accuracy')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

max_acc1 = max(val_acc1)
max_epoch = epochs[val_acc1.index(max_acc1)]
ax2.annotate(f'Peak: {max_acc1:.2f}% (Ep {max_epoch})', 
             xy=(max_epoch, max_acc1), 
             xytext=(max_epoch, max_acc1-5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             ha='center')

plt.tight_layout()
plt.show()