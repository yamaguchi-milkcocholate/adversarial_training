import unittest
from src.storage.datasetrepo import GTSRBRepository


class MyTestCase(unittest.TestCase):
    def test_gtsrb(self):
        GTSRBRepository.load_from_pickle()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
