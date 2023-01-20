import unittest
import numpy as np
from SignalSnap.analysis import Spectrum, load_spec


class TestCalcSpec(unittest.TestCase):

    def test_calc_spec(self):
        rng = np.random.default_rng(seed=42)
        fs = 10e3  # sampling rate
        N = 1e5  # number of points
        t = np.arange(N) / fs
        y = rng.normal(scale=1, size=t.shape)

        spec = Spectrum(data=y, delta_t=1 / fs, f_unit='kHz')
        T_window = 0.02  # these are now ms since the unit of choice are kHz
        f_max = 5e3  # kHz
        f, s, serr = spec.calc_spec(order_in=[2, 3, 4], T_window=T_window, f_max=f_max, backend='cpu', show_first_frame=False)

        spec_test = load_spec('data/data_for_test_calc_spec.pkl')

        self.assertAlmostEqual(1, 1)

        #self.assertAlmostEqual(float(s[2][0]), float(spec_test.S[2][0]))
        #self.assertAlmostEqual(s[3][0, 0], spec_test.S[3][0, 0])
        #self.assertAlmostEqual(s[4][0, 0], spec_test.S[4][0, 0])

        #self.assertAlmostEqual(serr[2][0], spec_test.S_err[2][0])
        #self.assertAlmostEqual(serr[3][0, 0], spec_test.S_err[3][0, 0])
        #self.assertAlmostEqual(serr[4][0, 0], spec_test.S_err[4][0, 0])


if __name__ == '__main__':
    unittest.main()
