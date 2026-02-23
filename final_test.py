from engine_for_finetuning import (
    final_test,
    merge,
    train_one_epoch,
    validation_one_epoch,
)

final_top1, final_top5 = merge('./checkpoints_rear2/', 1)
print(
    f"Accuracy of the network on the test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%"
)
log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top5}