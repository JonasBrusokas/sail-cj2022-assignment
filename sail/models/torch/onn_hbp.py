import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

from torch.nn.parameter import Parameter
import torch.nn.functional as F

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

    def __init__(self,
                 input_units: int,
                 output_units: int,
                 hidden_units: int,
                 n_hidden_layers: int = 1,
                 dropout: float = 0.2,
                 beta: float = 0.99,
                 learning_rate: float = 0.01,
                 smoothing: float = 0.2,
                 # batch_size: int = 32
                 ):
        super(_ONNHBPModel, self).__init__()

        self.input_units = input_units
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.n_hidden_layers = n_hidden_layers
        # self.batch_size = batch_size

        self.device = torch.device('cpu')

        self.beta = Parameter(torch.tensor(beta), requires_grad=False).to(self.device)
        self.learning_rate = Parameter(torch.tensor(learning_rate), requires_grad=False).to(self.device)
        self.smoothing = Parameter(torch.tensor(smoothing), requires_grad=False).to(self.device)

        self.loss_array = []

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_units, n_hidden_layers)] +
            [nn.Linear(n_hidden_layers, n_hidden_layers) for _ in range(self.n_hidden_layers - 1)])  #

        self.output_layers = nn.ModuleList([nn.Linear(n_hidden_layers, output_units) for i in range(
            self.n_hidden_layers)])  #

        self.alpha = Parameter(torch.Tensor(self.n_hidden_layers).fill_(1 / (self.n_hidden_layers + 1)),
                               requires_grad=False)  #

        self.do = nn.Dropout(p=dropout)
        self.actfn = nn.Tanh()
        self.dtype = torch.float  # ?

    def zero_grad(self):  #
        for i in range(self.n_hidden_layers):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def __update_alpha(self, losses_per_layer, l):  #
        self.alpha[l] *= torch.pow(self.beta, losses_per_layer)
        self.alpha[l] = torch.max(self.alpha[l], self.smoothing / self.n_hidden_layers)

    def __update_hidden_layer_weight(self, w, b, l):  #
        self.hidden_layers[l].weight.data -= self.learning_rate * w
        self.hidden_layers[l].bias.data -= self.learning_rate * b

    def update_weights(self, X, Y):

        batch_size, n_classes = Y.shape

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

        # NOTE: missing "show_loss"

        # TODO: is X really torch.tensor type?
    def forward(self, X: torch.Tensor):
        scores = self.predict_(X_data=X)
        return scores

    def forward_(self, X: torch.Tensor):

        # X = torch.from_numpy(X).float().to(self.device)
        x = F.relu(self.hidden_layers[0](X))

        hidden_connections = [x]
        for i in range (1, self.n_hidden_layers):
            hidden_connections += [
                F.relu(self.hidden_layers[i](hidden_connections[i - 1]))
            ]

        output_class = [self.output_layers[i](hidden_connections[i]) for i in range(self.n_hidden_layers)]

        pred_per_layer = torch.stack(output_class)
        return pred_per_layer

    # NOTE: we do not have the 'show_loss' here
    def partial_fit_(self, X_data, Y_data):
        self.validate_input_X(X_data)
        self.validate_input_Y(Y_data)
        self.update_weights(X_data, Y_data)

    def partial_fit(self, X_data, Y_data):
        self.partial_fit_(X_data, Y_data)

    # NOTE: this is basically CPU bound + output is numpy
    def predict_(self, X_data):
        scores = torch.sum(
            torch.mul(
                self.alpha.view(self.n_hidden_layers, 1).repeat(1, len(X_data))
                    .view(self.n_hidden_layers, len(X_data), 1)
                , self.forward_(X_data))
            , 0)
        # self.validate_input_X(X_data)
        return scores

            # .cpu().numpy()

    def predict(self, X_data):
        scores = self.predict_(X_data)
        return torch.argmax(scores, dim=1)

class ONNHBP_Classifier(NeuralNetClassifier):
    def __init__(self,
                 # in_channels, input_size, lstm_layers, classes,
                 input_units: int,
                 output_units: int,
                 hidden_units: int,
                 n_hidden_layers: int = 1,
                 dropout: float = 0.2,
                 beta: float = 0.99,
                 learning_rate: float = 0.01,
                 smoothing: float = 0.2,
                 ):
        """
        Online Neural Network trained with Hedge Backpropagation.
        Args:
            input_units: dimensionality of input
            output_units: number of classification classes
            hidden_units: number of hidden units in a single layer
            n_hidden_layers: number of layers in the network
            dropout: dropout coefficient
            beta: beta coefficient (b)
            learning_rate: learning rate for training
            smoothing: smoothing coefficient (s)
        """
        super(ONNHBP_Classifier, self).__init__(
            module=_ONNHBPModel,
            module__input_units=input_units,
            module__output_units=output_units,
            module__hidden_units=hidden_units,
            module__n_hidden_layers=n_hidden_layers,
            module__dropout=dropout,
            module__beta=beta,
            module__learning_rate=learning_rate,
            module__smoothing=smoothing,
            max_epochs=1,
        )
        self.train_split = None # Force disable splitting, might need to turn off later

if __name__ == '__main__':

    n_data_points = 40
    n_features = 15
    n_classes = 5

    classifier = ONNHBP_Classifier(
        input_units=n_features,
        output_units=n_classes,
        hidden_units=10,
        n_hidden_layers=7,
    )

    """
    #%%
        onn_network = ONN(features_size=10, max_num_hidden_layers=5, qtd_neuron_per_hidden_layer=40, n_classes=10)
        ##Creating Fake Classification Dataset
        X, Y = make_classification(n_samples=50000, n_features=10, n_informative=4, n_redundant=0, n_classes=10,
                                   n_clusters_per_class=1, class_sep=3)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)
    """

    from sklearn.datasets import make_classification
    import numpy as np

    def classification_data():
        X, y = make_classification(n_samples=n_data_points,
                                   n_features=n_features,
                                   n_informative=n_classes,
                                   random_state=0,
                                   n_classes=n_classes,
                                   n_clusters_per_class=1,
                                   )
        X, y = X.astype(np.float32), y.astype(np.int64)
        return X, y

    X_train, y_train = classification_data()

    # for i in range(len(X_train)):
    #     classifier.partial_fit(np.asarray([X_train[i, :]]), np.asarray([y_train[i]]))

    epochs = 5
    for i in range(epochs):
        classifier.partial_fit(X_train, y_train)


    train_losses = classifier.history[:, 'train_loss']
    # assert train_losses[0] > train_losses[-1]
    # valid_acc = classifier.history[-1, 'valid_acc']
