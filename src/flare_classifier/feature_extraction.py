import light_curve as lc
import numpy as np
import pandas as pd

amplitude = lc.Amplitude()
anderson_test = lc.AndersonDarlingNormal()
bazin_fit = lc.BazinFit(algorithm="mcmc")
beyond_1_std = lc.BeyondNStd(1.0)
beyond_2_std = lc.BeyondNStd(2.0)
cusum = lc.Cusum()
eta = lc.Eta()
eta_e = lc.EtaE()
excess_var = lc.ExcessVariance()
interp_range = lc.InterPercentileRange(quantile=0.25)
kurtosis = lc.Kurtosis()
mag_perc_ratio = lc.MagnitudePercentageRatio(quantile_numerator=0.4, quantile_denominator=0.05)
max_slope = lc.MaximumSlope()
mean = lc.Mean()
mean_var = lc.MeanVariance()
median = lc.Median()
median_abs_dev = lc.MedianAbsoluteDeviation()
median_buff_range_perc = lc.MedianBufferRangePercentage(quantile=0.1)
otsu_split = lc.OtsuSplit()
percent_ampl = lc.PercentAmplitude()
percent_diff_mag = lc.PercentDifferenceMagnitudePercentile(quantile=0.05)
skew = lc.Skew()
standard_dev = lc.StandardDeviation()
stetson_k = lc.StetsonK()
weighted_mean = lc.WeightedMean()

FEATURES_SET = [
    amplitude,
    anderson_test,
    bazin_fit,
    beyond_1_std,
    beyond_2_std,
    cusum,
    eta,
    eta_e,
    excess_var,
    interp_range,
    kurtosis,
    mag_perc_ratio,
    max_slope,
    mean,
    mean_var,
    median,
    median_abs_dev,
    median_buff_range_perc,
    otsu_split,
    percent_ampl,
    percent_diff_mag,
    skew,
    standard_dev,
    stetson_k,
    weighted_mean,
]

extractor = lc.Extractor(*FEATURES_SET)


def feature_extractor(dataframe, *, n_jobs=-1):
    t = dataframe["mjd"].to_numpy()
    mag = dataframe["mag"].to_numpy()
    magerr = dataframe["magerr"].to_numpy()

    light_curves = list(map(tuple, np.stack([t, mag, magerr], axis=1)))

    results = extractor.many(
        light_curves,
        n_jobs=n_jobs,
        sorted=True,
        check=False,
    )

    results_df = pd.DataFrame(results, columns=extractor.names)
    full_df = pd.concat([dataframe, results_df], axis=1)

    assert len(full_df) == len(dataframe)

    full_df.drop(["bazin_fit_amplitude"], axis=1, inplace=True)

    return full_df
