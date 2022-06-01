import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from skorch.classifier import NeuralNetClassifier


class _ONNHBPModel(nn.Module):
    """Hedge Backpropagation FFN model.
    Online Deep Learning: Learning Deep Neural Networks on the Fly https://arxiv.org/abs/1711.03705

    Args:
        input_units (int): Number of input units
        output_units (int): Number of output units
        hidden_units (int): Number of hidden units
        n_hidden_layers (int): Number of hidden layers
        dropout (float): Dropout

    """

    def __init__(self, input_units, output_units, hidden_units, n_hidden_layers=1, dropout=0.2, beta=0.99,
                 learning_rate=0.01, smoothing=0.2, batch_size=32):
        super(_RNNModel, self).__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.n_hidden_layers = n_hidden_layers

        self.beta = Parameter(torch.tensor(beta), requires_grad=False).to(self.device)
        self.learning_rate = Parameter(torch.tensor(n), requires_grad=False).to(self.device)
        self.smoothing = Parameter(torch.tensor(smoothing), requires_grad=False).to(self.device)

        self.loss_array = []

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_units, n_hidden_layers)] +
            [nn.Linear(n_hidden_layers, n_hidden_layers) for i in range(self.n_hidden_layers - 1)])

        self.output_layers = nn.ModuleList(
            [nn.Linear(n_hidden_layers, output_units) for i in range(self.n_hidden_layers)])

        self.alpha = Parameter(torch.Tensor(self.n_hidden_layers).fill_(1 / (self.n_hidden_layers + 1)),
                               requires_grad=False)

        self.do = nn.Dropout(p=dropout)
        self.actfn = nn.Tanh()
        self.device = torch.device('cpu')
        self.dtype = torch.float

    def zero_grad(self):
        for i in range(self.n_hidden_layers):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def __update_alpha(self, losses_per_layer, l):
        self.alpha[l] *= torch.pow(self.beta, losses_per_layer)
        self.alpha[l] = torch.max(self.alpha[l], self.smoothing / self.n_hidden_layers)

    def __update_hidden_layer_weight(w, b, l):
        self.hidden_layers[l].weight.data -= self.learning_rate * w
        self.hidden_layers[l].bias.data -= self.learning_rate * b

    def update_weights(self, X, Y):

        batch_size, n_classes = X.shape

        Y = torch.from_numpy(Y).to(self.device)

        predictions_per_layer = self.forward(X)

        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(out.view(batch_size, n_classes), Y.view(batch_size).long())
            losses_per_layer.append(loss)

        w = [0] * self.n_hidden_layers
        b = [0] * self.n_hidden_layers

        with torch.no_grad():

            for i in range(self.n_hidden_layers):
                losses_per_layer[i].backward(retain_graph=True)
                self.output_layers[i].weight.data -= self.learning_rate * self.alpha[i] * self.output_layers[
                    i].weight.grad.data
                self.output_layers[i].bias.data -= self.learning_rate * self.alpha[i] * self.output_layers[
                    i].bias.grad.data

                for j in range(i + 1):
                    w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                    b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()  # TODO: test torch.zero_grad()

            [self.__update_hidden_layer_weight(w[i], b[i], i) for i in range(self.n_hidden_layers)]

            [self.__update_alpha(i, losses_per_layer[i]) for i in range(self.n_hidden_layers)]

        z_t = torch.sum(self.alpha)

        self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

    def forward(self, X):
        X = torch.from_numpy(X).float().to(self.device)

        x = F.relu(self.hidden_layers[0](X))

        hidden_connections = [x] + [F.relu(self.hidden_layers[i](hidden_connections[i - 1])) for i in
                                    range(1, self.n_hidden_layers)]

        output_class = [self.output_layers[i](hidden_connections[i]) for i in range(self.n_hidden_layers)]

        pred_per_layer = torch.stack(output_class)

        return pred_per_layer

    def partial_fit_(self, X_data, Y_data):
        self.validate_input_X(X_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, Y_data, show_loss)

    def partial_fit(self, X_data, Y_data):
        self.partial_fit_(X_data, Y_data, show_loss)

    def predict_(self, X_data):
        self.validate_input_X(X_data)
        return torch.argmax(torch.sum(torch.mul(self.alpha.view(self.n_hidden_layers, 1).repeat(1, len(X_data)).view(
            self.n_hidden_layers, len(X_data), 1), self.forward(X_data)), 0), dim=1).cpu().numpy()

    def predict(self, X_data):
        pred = self.predict_(X_data)
        return pred


class RNNRegressor(NeuralNetRegressor):
    """Basic RNN/LSTM/GRU model.

    Args:
        input_units (int): Number of input units
        output_units (int): Number of output units
        hidden_units (int): Number of hidden units
        n_hidden_layers (int): Number of hidden layers
        output_nonlin (torch.nn.Module instance or None (default=nn.Linear)):
            Non-linearity to apply after last layer, if any.
        dropout (float): Dropout
        squeeze_output (bool): default=False
            Whether to squeeze output. Skorch requirement.
        cell_type (string): default="RNN"
        **kwargs: Arbitrary keyword arguments
    """

    def __init__(self, input_units, output_units, hidden_units,
                 n_hidden_layers=1, dropout=0.2, output_nonlin=nn.Linear, squeeze_output=False,
                 cell_type="RNN", **kwargs):
        super(RNNRegressor, self).__init__(
            module=_RNNModel,
            module__input_units=input_units,
            module__output_units=output_units,
            module__hidden_units=hidden_units,
            module__n_hidden_layers=n_hidden_layers,
            module__dropout=dropout,
            module__output_nonlin=output_nonlin,
            module__squeeze_output=squeeze_output,
            module__cell_type=cell_type,
            train_split=None, max_epochs=1, batch_size=20,
            **kwargs)