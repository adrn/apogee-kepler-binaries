from astropy.time import Time
from astropy.timeseries import BoxLeastSquares
import astropy.units as u
import numpy as np

from data_helpers import prepare_data


def get_eclipse(lc, P0=None, P_grid_size=10_000, P_grid=None,
                power_kwargs=None, n_eclipse=1):
    """Estimate the period of transits using BLS

    """

    if n_eclipse not in [1, 2]:
        raise ValueError("Number of eclipses, n_eclipse, can either "
                         f"be 1 or 2, not '{n_eclipse}'")

    # Convert to parts per thousand
    d = prepare_data(lc=lc)

    # We first run BLS to find the period and parameters of
    # the deepest eclipse:
    bls = BoxLeastSquares(d.flux_t,
                          d.flux_ppt,
                          dy=d.flux_ppt_err)

    if P0 is None and P_grid is None:  # default behavior
        period_grid = np.exp(np.linspace(np.log(1.), np.log(100),
                                         P_grid_size))

    elif P_grid is not None:
        period_grid = P_grid

    else:
        logP0 = np.log(P0.to_value(u.day))
        period_grid = np.exp(np.linspace(logP0 - 1, logP0 + 1,
                                         P_grid_size))

    if power_kwargs is None:
        power_kwargs = dict()
    power_kwargs.setdefault('duration', 0.1)
    power_kwargs.setdefault('oversample', 10)
    bls_power = bls.power(period_grid, **power_kwargs)

    # Find the highest power mode:
    index = np.argmax(bls_power.power)
    bls_period = bls_power.period[index] * u.day

    # Now at fixed period, we grid over duration:
    max_dur = min(bls_period.value / 2, 5.)
    duration_grid = np.logspace(-2, np.log10(max_dur), 256)
    bls_power_dur = bls.power(period=bls_power.period[index],
                              duration=duration_grid)
    bls_duration = bls_power_dur.duration[0] * u.day
    bls_t0 = bls_power_dur.transit_time[0]
    bls_depth = bls_power_dur.depth[0]

    transit_mask = bls.transit_mask(d.flux_t,
                                    period=bls_period.value,
                                    duration=1.5 * bls_duration.value,
                                    transit_time=bls_t0)

    eclipse1 = {
        'period': bls_period,
        't0': bls_t0,
        'astropy_t0': Time(bls_t0 + d.t_ref_jd, format='jd', scale='tdb'),
        'depth_ppt': bls_depth,
        'duration': bls_duration,
        'eclipse_mask': transit_mask
    }

    if n_eclipse == 1:
        return eclipse1

    # Mask out the first eclipse, and re-run on the masked data:
    m2 = ~transit_mask
    bls2 = BoxLeastSquares(d.flux_t[m2],
                           d.flux_ppt[m2],
                           dy=d.flux_ppt_err[m2])
    bls_power2 = bls2.power(bls_period.value, duration=duration_grid)
    bls_t02 = bls_power2.transit_time[0]
    bls_duration2 = bls_power2.duration[0] * u.day

    transit_mask = bls2.transit_mask(d.flux_t,
                                     period=bls_period.value,
                                     duration=1.5 * bls_duration2.value,
                                     transit_time=bls_t02)

    eclipse2 = {
        'period': bls_period,
        't0': bls_t02,
        'astropy_t0': Time(bls_t02 + d.t_ref_jd, format='jd', scale='tdb'),
        'depth_ppt': bls_power2.depth[0],
        'duration': bls_duration2,
        'eclipse_mask': transit_mask
    }

    return eclipse1, eclipse2
