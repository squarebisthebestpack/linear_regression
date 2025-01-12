import numpy as np 
from aprocA import reading, linear, exponent, polynomial, predict
import unittest

class tests(unittest.TestCase):
    def setUp(self):
        self.X1 = np.array([[1], [2], [3], [4]])
        #self.X2 = np.array([])
        self.y_l = np.array([2, 4, 6, 8])
        self.y_e = np.exp(self.y_l)
        self.y_p = np.array([2, 8, 18, 32])
        
    def test_linear(self):
       B = linear(self.X1, self.y_l)
       B_expected = [0, 2]
       np.testing.assert_almost_equal(B, B_expected, decimal=4)

    def test_exp(self):
        B = exponent(self.X1, self.y_e)
        B_expected = [0, 2]
        np.testing.assert_almost_equal(B, B_expected, decimal=4)

    def test_poly(self):
        B = polynomial(self.X1, self.y_p, 2)
        B_expected = [0, 0, 2]
        np.testing.assert_almost_equal(B, B_expected, decimal=4)

    def test_predict_linear(self):
        B = [0, 2]
        pred = predict(self.X1, B, '-p', None)
        pred_expected = [2, 4, 6, 8]
        np.testing.assert_almost_equal(pred, pred_expected, decimal=4)

    def test_predict_exp(self):
        B = [0, 2]
        pred = predict(self.X1, B, '-e', None)
        pred_expected = np.exp([2, 4, 6, 8])
        np.testing.assert_almost_equal(pred, pred_expected, decimal=4)
        

    def test_predict_poly(self):
        B = [0, 0, 2]
        pred = predict(self.X1, B, '-p', 2)
        pred_expected = [2, 8, 18, 32]
        np.testing.assert_almost_equal(pred, pred_expected, decimal=4)

    def test_exp_minus(self):
        y_minus = np.array([1,2,-3])
        self.assertRaises(SystemExit, exponent, self.X1, y_minus )
            
    def test_reading(self2):
        with open('test_data.csv', 'w') as data_expected:
            data_expected.write('x, y, z\n1, 2, 3\n4, 5, 6\n7, 8, 9')
        data = reading('test_data.csv')
        np.testing.assert_array_equal(data.values, np.array([[1,2,3],[4,5,6],[7,8,9]]))         
         
    def test_reading_empty(self2):
        with open('test_data.csv', 'w') as data_expected:
            data_expected.write('')
        with self2.assertRaises(SystemExit):  
            reading('test_data.csv')  



if __name__ == '__main__':
    unittest.main()        
        
