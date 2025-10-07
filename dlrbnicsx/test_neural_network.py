import unittest
import torch
import numpy as np
from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.neural_network.neural_network import HiddenLayersNet

class TestDataset(unittest.TestCase):
    """Unit test for Neural Network
    """
    def test_neural_network(self):
        """Checks shape of input, output
        and Neural Network parameters
        """
        input_parameter_set = np.ones([16, 7]).astype("f")

        model = HiddenLayersNet(input_parameter_set.shape[1],
                                [4, 22, 8, 90],
                                10, Tanh())
        model(input_parameter_set)
        print(f"Input data type: {type(model(input_parameter_set))}")

        model = HiddenLayersNet(input_parameter_set.shape[1],
                                [4, 22, 8, 90],
                                10, Tanh(),
                                return_numpy=True)
        ann_pred = model(input_parameter_set)
        print(f"Neural network prediction shape: {ann_pred.shape}")
        print(f"Neural network prediction type: {type(ann_pred)}")

        for param in model.parameters():
            print(f"Shape of NN parameter {param.data.shape} " +
                f"and dtype {param.data.dtype}")


if __name__ == "__main__":
    unittest.main()