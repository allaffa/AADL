import sys
import torch

import unittest

sys.path.append('../utils')
sys.path.append('../modules')
import dataloaders
from LinearRegression_models import LinearRegression
from optimizers import FixedPointIteration, RNA_Acceleration


###############################################################################


def linear_regression(slope, intercept, num_points, optimizer_str):
    input_dim, output_dim, dataset = dataloaders.linear_data(slope, intercept, num_points)
    use_bias = True
    learning_rate = 1e-3
    weight_decay = 0.0    
    batch_size = 1
    epochs = 100000
    threshold = 1e-9

    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    model = LinearRegression(input_dim, output_dim, use_bias)
    optimizer_classic = FixedPointIteration(dataloader, learning_rate, weight_decay)
    optimizer_classic.import_model(model)
    optimizer_classic.set_loss_function('mse')
    optimizer_classic.set_optimizer(optimizer_str)
    training_classic_loss_history = optimizer_classic.train(epochs, threshold, batch_size)
    
    weights = list(model.get_model().parameters())

    return weights[0].item(), weights[1].item(), training_classic_loss_history


def test_linear_regression(optimiser):
    num_points = 2
    straight_line_parameters = torch.rand(2, 1)
    slope = straight_line_parameters[0].item()
    intercept = straight_line_parameters[1].item()
    numeric_slope, numeric_intercept, history = linear_regression(slope, intercept, num_points, optimiser)
    print(optimiser+" converged in "+str(len(history))+" iterations "+"\n exact slope: "+str(slope)+"  - "+" numerical slope: "+str(numeric_slope)+"\n"+" exact intercept: "+str(intercept)+" - "+" numerical intercept: "+str(numeric_intercept))
    assert(abs((slope-numeric_slope))<1e-3 and abs((intercept-numeric_intercept))<1e-3)



###############################################################################


class TestLinearRegression(unittest.TestCase):

    def test_sgd(self):
        test_linear_regression('sgd')
    
    def test_rmsprop(self):
        test_linear_regression('rmsprop')
    
    def test_adam(self):
        test_linear_regression('adam')

###############################################################################


if __name__ == "__main__":
    unittest.main()

