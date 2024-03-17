import unittest
from data_loader import GetLoader
import torch

class TestGetLoader(unittest.TestCase):
    def setUp(self):
        self.data_root = "X_kla120.mat"
        self.data_label_root = "EQvec_kla120.mat"
        self.transform = None
        self.loader = GetLoader(self.data_root, self.data_label_root, self.transform)
        
    def test_data_loading(self):
        # Check if the data and labels are loaded correctly
        self.assertIsNotNone(self.loader.data)
        self.assertIsNotNone(self.loader.data_label)
        
    def test_transform(self):
        # Check if the data is transformed correctly
        transformed_data = self.loader.transform(self.loader.data)
        self.assertIsNotNone(transformed_data)
        
    def test_get_item(self):
        # Check if __getitem__ returns the correct data and label tensors
        item = 0
        data, label = self.loader.__getitem__(item)
        self.assertIsNotNone(data)
        self.assertIsNotNone(label)
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        
    def test_len(self):
        # Check if __len__ returns the correct length of the dataset
        length = self.loader.__len__()
        self.assertIsNotNone(length)
        self.assertIsInstance(length, int)
        self.assertEqual(length, len(self.loader.data))
        
if __name__ == '__main__':
    unittest.main()