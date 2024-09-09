import torch
import tqdm
import onnx2torch

class GetReshapeInit(torch.fx.Interpreter):
    def call_module(self, target, args, kwargs):
        print(type(target), target)
        print(type(args), args)
        print(type(kwargs), kwargs)

        return super().call_module(n)

class ModelTrainer:
    def __init__(self, max_epochs, learning_rate, device):
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.device = device

    def before_train_do(self, model):

        self.remove_handle = []

        def pre_hook(module, args):
            modified_shape = torch.tensor([args[0].size(0), -1])
            return (args[0], modified_shape)

        # This is to fix batch size in sri_resnet_a
        for name, module in model.named_modules():
            if isinstance(module, onnx2torch.node_converters.reshape.OnnxReshape):
                self.remove_handle.append(module.register_forward_pre_hook(pre_hook))

    def after_train_do(self, model):

        for handle in self.remove_handle:
            handle.remove()

    def train(self, model, data_loader, l1_reg=0, l2_reg=0):

        #optimiser = torch.optim.Adam(params=model.parameters(), weight_decay=1e-4, lr=1e-3)
        optimiser = torch.optim.SGD(params=model.parameters(), lr=self.learning_rate)

        self.before_train_do(model)

        for epoch in range(self.max_epochs):

            print(f"Training epoch: {epoch + 1}/{self.max_epochs}")
            iter_train = iter(data_loader)

            num_correct = 0
            num_samples = 0

            pbar = tqdm.tqdm(range(len(data_loader)))
            pbar.set_description(f"Training Accuracy: {0}")

            for _ in pbar:

                x, y = next(iter_train)

                model.train()
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = torch.nn.functional.log_softmax(model(x), dim=1)

                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
                train_acc = num_correct / num_samples
                pbar.set_description(f"Epoch Accuracy: {train_acc:.2f}")

                l1_regularization_loss = 0
                l2_regularization_loss = 0
                for param in model.parameters():
                    l1_regularization_loss += torch.sum(torch.abs(param))
                    l2_regularization_loss += torch.sum(param ** 2)

                loss = (torch.nn.functional.cross_entropy(scores, y) +
                        l1_reg * l1_regularization_loss +
                        l2_reg * l2_regularization_loss)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        self.after_train_do(model)

        return (num_correct / num_samples).item()

    def calculate_score(self, model, data_loader):

        num_correct = 0
        num_samples = 0

        model.eval()

        self.before_train_do(model)

        with torch.no_grad():
            for x, y in data_loader:

                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            acc = float(num_correct) / num_samples

        self.after_train_do(model)

        return acc

