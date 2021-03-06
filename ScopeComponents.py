class LiquidMedium(object):
    name = None
    RI = None

    def getName(self):
        return self.name

    def getRI(self):
        return self.RI


class Water(LiquidMedium):
    name = 'Water'
    RI = 1.33


class Oil(LiquidMedium):
    name = 'Oil'
    RI = 1.518


class ObjectiveLens(object):
    title = None
    numericalAperture = None
    magnification = None
    immersionMediumRI = None

    def get_numerical_aperture(self):
        return self.numericalAperture

    def get_immersion_refractive_index(self):
        return self.immersionMediumRI

    def get_magnification(self):
        return self.magnification

    def get_title(self):
        return self.title


class WaterObjectiveLens60x(ObjectiveLens):
    title = 'Water Objective Lens Plan'
    numericalAperture = 1.2
    magnification = 60
    immersionMediumRI = Water().getRI()


class OilObjectiveLens60x(ObjectiveLens):
    title = 'Oil objective lens Apo'
    numericalAperture = 1.49
    magnification = 60
    immersionMediumRI = Oil().getRI()


class Sample(object):
    coverslip = None
    mountingMediumRI = None

    def get_mounting_medium_RI(self):
        return self.mountingMediumRI


class StandardSample(Sample):
    coverslip = "1.5H"
    mountingMediumRI = 1.33


class PixelSizeCalibrationUm(object):
    micronsPerPixel = None

    def __set__(self, micronsPerPixel):
        self.micronsPerPixel = micronsPerPixel

    def __get__(self):
        return self.micronsPerPixel


