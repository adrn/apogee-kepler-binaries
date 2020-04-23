from dataclasses import dataclass

import astropy.units as u
import numpy as np

BKJD_OFFSET = 2454833.


@dataclass
class RawData:
    """TODO"""
    t_ref_jd: float

    flux_t: np.ndarray = None
    flux_ppt: np.ndarray = None
    flux_ppt_err: np.ndarray = None

    rv_t: np.ndarray = None
    rv_mps: np.ndarray = None
    rv_mps_err: np.ndarray = None

    def plot(self, axes=None, P=None, t0=None, **kwargs):
        import matplotlib.pyplot as plt

        fold = False
        if t0 is not None and P is None:
            raise ValueError()

        elif P is not None:
            fold = True

            if t0 is None:
                t0 = 0.

        if axes is None:
            if fold:
                fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                                         sharex=True)

            else:
                fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        else:
            fig = axes[0].figure

        # Default plot styling:
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('color', 'k')
        kwargs.setdefault('ecolor', '#aaaaaa')

        default_mew = kwargs.pop('mew', 0)
        kwargs.setdefault('markeredgewidth', default_mew)

        default_ms = kwargs.pop('ms', 2.)
        kwargs.setdefault('markersize', default_ms)

        default_ls = kwargs.pop('ls', 'none')
        kwargs.setdefault('linestyle', default_ls)

        if fold:
            flux_x = ((self.flux_t - t0) / P + 0.5) % 1 - 0.5
            rv_x = ((self.rv_t - t0) / P + 0.5) % 1 - 0.5
            xlabel = 'phase'

            for i in [-1, 0, 1]:
                axes[0].errorbar(flux_x + i, self.flux_ppt,
                                 **kwargs)
                axes[1].errorbar(rv_x + i, self.rv_mps,
                                 yerr=self.rv_mps_err,
                                 **kwargs)

            axes[0].set_xlim(-1, 1)

        else:
            flux_x = self.flux_t
            rv_x = self.rv_t
            xlabel = 'time [days]'

            axes[0].errorbar(flux_x, self.flux_ppt,
                             **kwargs)
            axes[1].errorbar(rv_x, self.rv_mps,
                             yerr=self.rv_mps_err,
                             **kwargs)

        axes[1].set_xlabel(xlabel)
        axes[0].set_ylabel('flux [ppt]')
        axes[1].set_ylabel(f'RV [{u.m/u.s:latex_inline}]')

        return fig

    @property
    def has_lc(self):
        return self.flux_t is not None and self.flux_ppt is not None

    @property
    def has_rv(self):
        return self.rv_t is not None and self.rv_mps is not None


def sigma_clip_lc(lc, sigma=5, return_mask=False, positive_only=True):
    MAD = np.nanmedian(np.abs(lc.flux - np.nanmedian(lc.flux)))

    if positive_only:
        sigma_mask = (lc.flux - 1) < (sigma * 1.5 * MAD)
    else:
        sigma_mask = np.abs(lc.flux - 1) < (sigma * 1.5 * MAD)

    if return_mask:
        return sigma_mask

    else:
        return lc[sigma_mask]


def prepare_data(lc=None, rv=None):
    """Strip units, filter bad values, and put data into the
    same time system.

    Parameters
    ----------
    lc : `lightkurve.KeplerLightCurve` (optional)
    rv : `thejoker.RVData` (optional)

    Returns
    -------
    ??
    """

    if lc is None and rv is None:
        raise ValueError("You must either specify a light curve, "
                         "or radial velocity data! (or both)")

    data_kw = {}
    t_ref = None

    if lc is not None:

        # Convert to parts per thousand
        mask = np.isfinite(lc.flux) & np.isfinite(lc.flux_err)
        y = (lc.flux[mask] / np.median(lc.flux[mask]) - 1) * 1e3
        yerr = lc.flux_err[mask] * 1e3

        t = lc.astropy_time.tcb.jd[mask]
        t_ref = BKJD_OFFSET
        t = t - t_ref

        data_kw['flux_t'] = t
        data_kw['flux_ppt'] = y
        data_kw['flux_ppt_err'] = yerr

    if rv is not None:
        if t_ref is None:
            t_ref = rv.t.tcb.jd

        data_kw['rv_t'] = rv.t.tcb.jd - t_ref
        data_kw['rv_mps'] = rv.rv.to_value(u.m/u.s)
        data_kw['rv_mps_err'] = rv.rv_err.to_value(u.m/u.s)

    data_kw['t_ref_jd'] = t_ref

    return RawData(**data_kw)
