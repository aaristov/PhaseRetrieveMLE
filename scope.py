import numpy as np
from Zernike import PupilFunction,PupilMask,FitPupilFunction,ZernikePolynomials
from ScopeComponents import *
import logging
logging.basicConfig(level=logging.DEBUG)

class Scope(object):
    parameters=dict(numerical_aperture=None,
                    pixel_size_um=None,
                    immersion_medium_refractive_index=None,
                    mounting_medium_RI=None)

    def __init__(self, pixel_calibration:PixelSizeCalibrationUm, objective_lens:ObjectiveLens=OilObjectiveLens60x(), sample:Sample=StandardSample()):
        self._add_objective_lens(objective_lens)
        self._add_sample(sample)
        self._add_pixel_calibration(pixel_calibration)
        logging.info('Scope parameters set: {}'.format(self.parameters))
        self.pupilFunc = None
        self.zernikePolynomials = None
        self.pupilMask = None

    def _add_objective_lens(self, objective_lens:ObjectiveLens):
        NA = objective_lens.get_numerical_aperture()
        assert isinstance(NA,float)
        self.parameters['numerical_aperture']=NA

        RI = objective_lens.get_immersion_refractive_index()
        assert isinstance(RI,float)
        self.parameters['immersion_medium_refractive_index']=RI

    def _add_sample(self, sample:Sample):
        RI = sample.get_mounting_medium_RI()
        assert isinstance(RI,float)
        self.parameters["mounting_medium_RI"]=RI


    def _add_pixel_calibration(self, px:PixelSizeCalibrationUm):
        px_ = px
        assert isinstance(px_,float)
        self.parameters["pixel_size_um"]=px_

    def init_pupil(self):
        self.pupilMask = PupilMask(**self.parameters)
        self.zernikePolynomials = ZernikePolynomials(self.pupilMask, 15)
        self.pupilFunc = PupilFunction(self.zernikePolynomials, self.pupilMask)

    def get_frame(self):
        pass

    def calibrate_pupil(self):
        pass

    def get_localizations(self):
        pass


class CalibratePupilFuction(Scope):
    pupil_fitter=NotImplemented

    def __init__(self, **parameters_dict):
        self.initialize_fitter()
        self.parameters

    def initialize_fitter(self)->pupil_fitter:
        self.init_pupil_mask()
        self.init_pupil_function()
        self.init_pupil_fitter()

    def fit(self,z_stack:np.ndarray,z_vector:np.ndarray):
        return self.pupil_fitter(z_stack,z_vector)

    def init_pupil_mask(self)->PupilMask:
        raise NotImplementedError

    def init_pupil_function(self)->PupilFunction:
        raise NotImplementedError

    def init_pupil_fitter(self)->FitPupilFunction:
        raise NotImplementedError

