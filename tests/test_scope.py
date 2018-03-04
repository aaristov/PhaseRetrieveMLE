import unittest
from Zernike import PupilMask,ZernikePolynomials,PupilFunction
from scope import Scope
import ScopeComponents as SC
import numpy as np

class TestScope(unittest.TestCase):

    def setUp(self):
        sample = SC.StandardSample()
        obj = SC.WaterObjectiveLens60x()
        px = SC.PixelSizeCalibrationUm()
        px = 0.11
        self.scope = Scope(px,obj,sample)

    def test_scope_integrity(self):
        self.assertEqual(self.scope.parameters['mounting_medium_RI'],1.33)
        self.assertEqual(self.scope.parameters['immersion_medium_refractive_index'],1.33)
        self.assertEqual(self.scope.parameters['pixel_size_um'],0.11)
        self.assertEqual(self.scope.parameters['numerical_aperture'],1.2)

    def test_scope_pupilFunc(self):
        self.scope.init_pupil()
        self.assertEqual(self.scope.pupilMask.NA,1.2)

        self.pupilFunc = self.scope.pupilFunc
        psf = self.pupilFunc.gen_PSF(0,0,0,100,0,32)
        self.assertEqual(psf.shape,(32,32),'Shape is wrong')
        self.assertAlmostEqual(psf.sum(),100,delta=1.,msg='Intensity is wrong')




if __name__ == '__main__':
    unittest.main()
