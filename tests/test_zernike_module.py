import unittest
from Zernike import PupilMask,ZernikePolynomials,PupilFunction
import numpy as np

class TestPupil(unittest.TestCase):

    def setUp(self):
        self.pupil = PupilMask()

    def test_if_parabola_correct_size(self):
        self.assertEqual(self.pupil.img_size,len(self.pupil.parabola))

class TestZernikePolynomials(unittest.TestCase):

    def setUp(self):
        self.pupil = PupilMask()
        self.zernike = ZernikePolynomials(self.pupil, 15)

    def test_zernike_indices(self):
        self.assertEqual(len(self.zernike.zernInd), self.zernike.num, 'number of Zernike Indices is good')

    def test_zernike_stack(self):
        self.assertEqual(len(self.zernike.zernStack), self.zernike.num, 'number of Zernike slices is good')

    @unittest.skip('skipping plot test')
    def test_zernike_plot(self):
        self.zernike.plotZern()

    def test_wrong_zernike_number(self):
        with self.assertRaises(AssertionError):
            zz=ZernikePolynomials(self.pupil, 'abc')

    def test_zero_zernike_number(self):
        with self.assertRaises(AssertionError):
            zz=ZernikePolynomials(self.pupil, 0)

    def test_negative_zernike_number(self):
        with self.assertRaises(AssertionError):
            zz=ZernikePolynomials(self.pupil, -10)

    def test_floating_zernike_number(self):
        with self.assertRaises(AssertionError):
            zz=ZernikePolynomials(self.pupil, np.random.rand())


class TestPupilFunction(unittest.TestCase):

    def setUp(self):
        self.p = PupilMask()
        self.z = ZernikePolynomials(self.p,15)
        self.pupilFunc = PupilFunction(self.z,self.p)

    def test_init(self):
        self.assertTrue(np.array_equal(self.pupilFunc.getWeights(),np.zeros(15+2)))

    def test_gen_PSF(self):
        psf = self.pupilFunc.gen_PSF(0,0,0,100,0,32)
        self.assertEqual(psf.shape,(32,32),'Shape is wrong')
        self.assertAlmostEqual(psf.sum(),100,delta=1.,msg='Intensity is wrong')

    def test_getWeights(self):
        self.assertTrue(np.array_equal(self.pupilFunc.getWeights(),np.zeros(15+2)))

    def test_updateWeight(self):
        self.pupilFunc.updateWeight(10,1)
        target = np.zeros(15 + 2)
        target[10]+=1
        self.assertTrue(np.array_equal(self.pupilFunc.getWeights(), target))



if __name__ == '__main__':
    unittest.main()
