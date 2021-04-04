#!/usr/bin/python

import sys
import unittest

from data_processing import DataProcessing
from model import ModelPredict, ModelTrain


class TestStringMethods(unittest.TestCase):

    def test_dataset(self):
        dp = DataProcessing()
        dataset = dp.get_dataframe_to_train()
        self.assertGreater(dataset.shape[0], 50)

    def test_model_train(self):
        dp = DataProcessing()
        dataset = dp.get_dataframe_to_train()
        model_predict = ModelPredict()
        self.assertEqual(model_predict.predict(10).shape[0], 10)


if __name__ == '__main__':
    unittest.main()
