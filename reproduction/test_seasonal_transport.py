"""Analytical checks for projected empirical transport representations."""
import unittest
import numpy as np
from seasonal_transport import projected_quantiles, monthly_flow_quantiles

class ProjectedTransportTests(unittest.TestCase):
    def test_monthly_scale_invariance_and_zero_flow(self):
        months=np.repeat(np.arange(1,13),4)
        q=np.tile([0.,0.,2.,5.],12)
        np.testing.assert_allclose(monthly_flow_quantiles(q,months),
                                   monthly_flow_quantiles(q*100,months))
        self.assertEqual(monthly_flow_quantiles(q,months)[0,0],0.)

    def test_monthly_permutation_invariance_and_seasonal_sensitivity(self):
        months=np.repeat(np.arange(1,13),4)
        q=np.arange(48,dtype=float)+1
        first=monthly_flow_quantiles(q,months)
        shuffled=q.reshape(12,4)[:,::-1].ravel()
        np.testing.assert_array_equal(first,monthly_flow_quantiles(shuffled,months))
        shifted=monthly_flow_quantiles(np.roll(q,24),months)
        self.assertGreater(float(np.mean((first-shifted)**2)),.01)
        np.testing.assert_allclose(first,np.roll(shifted,6,axis=0))

    def test_constant_flow_months_coincide(self):
        months=np.repeat(np.arange(1,13),4)
        np.testing.assert_allclose(monthly_flow_quantiles(np.ones(48),months),np.log(2))

    def test_one_dimensional_translation(self):
        points=np.array([[0.],[1.],[3.],[6.]])
        weights=np.ones(4)/4
        first=projected_quantiles(points,weights,np.ones((1,1)),4)
        second=projected_quantiles(points+2,weights,np.ones((1,1)),4)
        self.assertAlmostEqual(float(np.sqrt(np.mean((first-second)**2))),2)

    def test_mass_splitting_does_not_change_measure(self):
        first=projected_quantiles(np.array([[0.],[4.]]),np.array([.25,.75]),np.ones((1,1)),4)
        second=projected_quantiles(np.array([[4.],[0.],[4.]]),np.array([.5,.25,.25]),np.ones((1,1)),4)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first,[[0.,4.,4.,4.]])

    def test_calendar_wrap_is_continuous(self):
        angle=np.array([2*np.pi/365,2*np.pi*364/365,np.pi])
        points=np.column_stack([np.cos(angle),np.sin(angle),np.zeros(3)])
        self.assertLess(np.linalg.norm(points[0]-points[1]),.04)
        self.assertGreater(np.linalg.norm(points[0]-points[2]),1.9)

if __name__=='__main__':unittest.main()
