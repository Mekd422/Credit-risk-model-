import unittest
import pandas as pd
from pathlib import Path
from src.data_processing import process_and_save_data


class TestDataProcessing(unittest.TestCase):

    def test_is_high_risk_exists(self):
        # Create dummy data
        data = {
            "CustomerId": ["C1", "C2", "C3"],
            "TransactionId": ["T1", "T2", "T3"],
            "TransactionStartTime": ["2025-12-01", "2025-12-02", "2025-12-03"],
            "Amount": [100, 200, 150],
            "ProviderId": ["P1", "P2", "P3"],
            "ProductCategory": ["cat1", "cat2", "cat3"],
            "ChannelId": ["ch1", "ch2", "ch3"],
            "PricingStrategy": [1, 2, 2],
            "BatchId": ["B1", "B2", "B3"],
            "AccountId": ["A1", "A2", "A3"],
            "SubscriptionId": ["S1", "S2", "S3"],
            "CurrencyCode": ["UGX", "UGX", "UGX"],
            "CountryCode": [256, 256, 256],
            "Value": [100, 200, 150],
            "FraudResult": [0, 0, 0],
        }

        df = pd.DataFrame(data)

        # Save to temporary CSV
        tmp_path = Path("./tmp_test.csv")
        df.to_csv(tmp_path, index=False)

        # Process the data
        output_path = Path("./tmp_processed.csv")
        final_df = process_and_save_data(str(tmp_path), str(output_path))

        # Test if 'is_high_risk' column exists
        self.assertIn("is_high_risk", final_df.columns)

        # Clean up
        tmp_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
