import unittest
from Zernike import Pupil,Zernike
import numpy as np

class TestPupil(unittest.TestCase):

    def setUp(self):
        self.pupil = Pupil()

    def test_if_parabola_correct_size(self):
        self.assertEqual(self.pupil.img_size,len(self.pupil.parabola))

class TestZernikeObject(unittest.TestCase):

    def setUp(self):
        self.pupil = Pupil()
        self.zernike = Zernike(self.pupil,15)

    def test_zernike_indices(self):
        self.assertEqual(len(self.zernike.zernInd), self.zernike.num, 'number of Zernike Indices is good')

    def test_zernike_stack(self):
        self.assertEqual(len(self.zernike.zernStack), self.zernike.num, 'number of Zernike slices is good')

    @unittest.skip('skipping plot test')
    def test_zernike_plot(self):
        self.zernike.plotZern()

    def test_wrong_zernike_number(self):
        with self.assertRaises(ValueError):
            zz=Zernike(self.pupil,'abc')

    def test_zero_zernike_number(self):
        with self.assertRaises(ValueError):
            zz=Zernike(self.pupil,0)

    def test_negative_zernike_number(self):
        with self.assertRaises(ValueError):
            zz=Zernike(self.pupil,-10)

    def test_floating_zernike_number(self):
        with self.assertRaises(ValueError):
            zz=Zernike(self.pupil,np.random.rand())


if __name__ == '__main__':
    unittest.main()
