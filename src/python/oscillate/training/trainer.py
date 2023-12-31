import torch
from tqdm import tqdm


class TTATrainer:

    def __init__(self, model, loss_function, optimizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Found use", torch.cuda.device_count(), "GPUs.")
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        self.loss_function = loss_function
        self.optimizer = optimizer

    def summary(self):
        print("Model Summary")
        print("Layer Name" + "\t" * 7 + "Number of Parameters")
        print("=" * 100)
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            total_params += param
            print(name + "\t" * 3 + str(param))
        print("=" * 100)
        print(f"Total Params:{total_params}")

    def train(self, dataloader, val_dataloader=None, epochs: int = 1, progress: bool = False):
        self.summary()

        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(dataloader) if progress else dataloader
            for X_enc, X_dec, y in pbar:
                X_enc, X_dec, y = X_enc.to(self.device), X_dec.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X_enc, X_dec)
                loss = self.loss_function(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if progress:
                    pbar.set_description(f"Epoch {epoch + 1} loss: {running_loss}")
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch + 1} completed, loss: {epoch_loss}")
            train_losses.append(epoch_loss)

            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                val_losses.append(val_loss)
                print(f"Validation loss: {val_loss}")

        return train_losses, val_losses

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_enc, X_dec, y in dataloader:
                X_enc, X_dec, y = X_enc.to(self.device), X_dec.to(self.device), y.to(self.device)
                y_hat = self.model(X_enc, X_dec)
                loss = self.loss_function(y_hat, y)
                total_loss += loss.item()
        return total_loss / len(dataloader)

