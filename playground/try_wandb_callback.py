from pipeline import callbacks

wandb_logger_callback = callbacks.get_wandb_logger_callback()

print(wandb_logger_callback)