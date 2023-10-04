import torch
from tqdm import tqdm


class TTATrainer:

    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self, dataloader, epochs: int = 1, progress: bool = False):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(dataloader) if progress else dataloader
            for X_enc, X_dec, y in pbar:
                self.optimizer.zero_grad()
                y_hat = self.model(X_enc, X_dec)
                loss = self.loss_function(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if progress:
                    pbar.set_description(f"Epoch {epoch + 1} loss: {running_loss}")
            print(f"Epoch {epoch + 1} completed, loss: {running_loss / len(dataloader)}")

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(dataloader)
