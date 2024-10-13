from scipy.optimize import curve_fit
import numpy as np
import pickle

def fit_biexponential(tau_timeseries, ac_timeseries):
    """Fit a biexponential function to a hydrogen bond time autocorrelation function
    Return the two time constants
    """
    def model(t, A, tau1, B, tau2):
        """Fit data to a biexponential function.
        """
        return A * np.exp(-t / tau1) + B * np.exp(-t / tau2)

    params, params_covariance = curve_fit(model, tau_timeseries, ac_timeseries, [1, 0.5, 1, 2])

    fit_t = np.linspace(tau_timeseries[0], tau_timeseries[-1], 1000)
    fit_ac = model(fit_t, *params)

    return params, fit_t, fit_ac

read_file = open('HB_lifetimes.out','rb')
tau_frames = pickle.load(read_file)
f_out = open('HB_lifetime_constants.out','w')

#P-P
HB_lifetimes = pickle.load(read_file)
if np.isnan(np.sum(HB_lifetimes)) == False:
    params, fit_t, fit_ac = fit_biexponential(tau_frames, HB_lifetimes)

    A, tau1, B, tau2 = params
    time_constant = A * tau1 + B * tau2
    f_out.write('pp_tau = {}\n'.format(time_constant))

#Pa-Wd
HB_lifetimes = pickle.load(read_file)   
if np.isnan(np.sum(HB_lifetimes)) == False:
    params, fit_t, fit_ac = fit_biexponential(tau_frames, HB_lifetimes)

    A, tau1, B, tau2 = params
    time_constant = A * tau1 + B * tau2
    f_out.write('pa_wd_tau = {}\n'.format(time_constant))

#Pd-Wa
HB_lifetimes = pickle.load(read_file)   
if np.isnan(np.sum(HB_lifetimes)) == False:
    params, fit_t, fit_ac = fit_biexponential(tau_frames, HB_lifetimes)

    A, tau1, B, tau2 = params
    time_constant = A * tau1 + B * tau2
    f_out.write('pd_wa_tau = {}\n'.format(time_constant))

#W-W
HB_lifetimes = pickle.load(read_file)   
if np.isnan(np.sum(HB_lifetimes)) == False:
    params, fit_t, fit_ac = fit_biexponential(tau_frames, HB_lifetimes)

    A, tau1, B, tau2 = params
    time_constant = A * tau1 + B * tau2
    f_out.write('ww_tau = {}\n'.format(time_constant))
f_out.close()