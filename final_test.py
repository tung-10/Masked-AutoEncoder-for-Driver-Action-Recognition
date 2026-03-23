from engine_for_finetuning import (
    final_test,
    merge,
    train_one_epoch,
    validation_one_epoch,
    merge_isolated,
)

class_acc, final_top1, final_top5 = merge_isolated('./checkpoints_front2/', 1)
print(
    f"Accuracy of the network on the test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%"
)
print("Per-class accuracies:")
for i, acc in enumerate(class_acc):
    print(f"  Class {i}: {acc:.2f}%")
log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top5}