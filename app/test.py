#!/usr/bin/python

import sys
import unittest

from data_processing import DataProcessing
from model import ModelPredict, ModelTrain


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.data = DataProcessing()
        self.data.load_datasets()

    def test_X_train(self):
        self.assertIsNotNone(self.data.X_train)

    def test_y_train(self):
        self.assertIsNotNone(self.data.y_train)

    def test_X_test(self):
        self.assertIsNotNone(self.data.X_test)

    def test_y_test(self):
        self.assertIsNotNone(self.data.y_test)

    def test_X_val(self):
        self.assertIsNotNone(self.data.X_val)

    def test_y_val(self):
        self.assertIsNotNone(self.data.y_val)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = ModelPredict()

    def test_prediction(self):
        self.assertGreater(self.model.predict("person1946_bacteria_4874.jpeg") > 0.5)


if __name__ == "__main__":
    unittest.main()
