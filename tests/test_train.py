import unittest
import pandas as pd
from src.train import load_data
import os


class TestTrain(unittest.TestCase):

    def test_load_data(self):
        df = pd.DataFrame({
            "CustomerId": ["C1", "C2", "C3"],
            "is_high_risk": [0, 1, 0],
            "feature1": [10, 20, 30],
            "feature2": [5, 3, 6],
        })

        df.to_csv("tmp.csv", index=False)

        X, y = load_data("tmp.csv")

        self.assertIn("feature1", X.columns)
        self.assertEqual(len(y), 3)

        os.remove("tmp.csv")


if __name__ == "__main__":
    unittest.main()
