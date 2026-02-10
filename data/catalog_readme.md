# ZTF DR17 M-dwarf Flares Catalog

Catalog of stellar flares detected in ZTF DR17 data for M-dwarf stars.

## Data File

- `flares_catalog.csv` — Main catalog file

## Column Descriptions

| Column | Unit | Description |
|--------|------|-------------|
| `oid` | — | ZTF Object ID |
| `ra` | deg | Right Ascension (J2000) |
| `dec` | deg | Declination (J2000) |
| `distance` | pc | Geometric distance from Gaia EDR3 (Bailer-Jones et al. 2021) |
| `distance_low` | pc | Lower bound of distance (16th percentile) |
| `distance_up` | pc | Upper bound of distance (84th percentile) |
| `plx_flag` | — | Parallax quality flag: 0 = good (RPlx ≥ 5), 1 = poor (RPlx < 5) |
| `R_r` | — | Reddening vector for r-band: 2.271 (SFD) or 2.617 (Bayestar) |
| `A_r` | mag | Extinction in r-band |
| `A_r_low` | mag | Lower bound of A_r |
| `A_r_up` | mag | Upper bound of A_r |
| `peak_time` | day | Flare peak time (MJD − 58000) |
| `fwhm` | hour | Full Width at Half Maximum of the flare |
| `amplitude` | mag | Flare amplitude in r-band |
| `num_points` | — | Number of photometric points in the flare |
| `spec_class` | — | Spectral class (M0–M7) from Gaia or Pan-STARRS |
| `energy` | erg | Bolometric flare energy |
| `energy_low` | erg | Lower bound of bolometric energy |
| `energy_up` | erg | Upper bound of bolometric energy |

## Notes

- **Distance**: Geometric distances from Gaia EDR3 using the probabilistic approach of Bailer-Jones et al. (2021)
- **Extinction**: Derived from SFD (Schlegel et al. 1998) or Bayestar2019 (Green et al. 2019) dust maps
- **Energy**: Calculated using the Mendoza flare model. 
- **Spectral class**: Preferentially from Gaia spectra, otherwise from Pan-STARRS photometry

## References

- Gaia Collaboration et al. (2023), A&A, 674, A1 - Gaia DR3
- Chambers et al. (2016), arXiv:1612.05560 - Pan-STARRS1 Surveys
- Bailer-Jones et al. (2021), AJ, 161, 147 - Gaia EDR3 distances
- Green et al. (2019), ApJ, 887, 93 - Bayestar2019 dust map
- Schlafly & Finkbeiner (2011), ApJ, 737, 103 - SFD recalibration
- Schlegel, Finkbeiner & Davis (1998), ApJ, 500, 525 - SFD dust map
- Bellm et al. (2019), PASP, 131, 018002 - ZTF survey

## Citation

If you use this catalog, please cite: https://doi.org/10.1093/mnras/stag145


