from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, model_name):
        self.writer = SummaryWriter(f"logs/{model_name}")
    
    def log_metrics(self, epoch, train_loss, train_acc, val_acc):
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_acc, epoch)
        self.writer.add_scalar("Accuracy/val", val_acc, epoch)
    
    def close(self):
        self.writer.close()