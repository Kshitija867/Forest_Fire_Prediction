import numpy as np

def calculate_fwi_indices(weather):
    """Computes structural FWI tracking codes using meteorological equations."""
    temp, rh, ws, rain = weather["temp"], weather["rh"], weather["ws"], weather["rain"]
    
    # Standard historical baseline assumptions
    ffmc_yesterday, dmc_yesterday, dc_yesterday = 85.0, 24.0, 100.0
    
    # FFMC calculation
    ed = 0.942 * (rh**0.679) + (11.0 * np.exp((rh - 100) / 10)) + (0.18 * (100 - rh)) * (1.0 - np.exp(-0.115 * ws))
    moisture_content = 147.2 * (101.0 - ffmc_yesterday) / (59.5 + ffmc_yesterday)
    ko = 0.424 * (1.0 - (rh / 100)**1.7) + 0.0694 * (ws**0.5) * (1.0 - (rh / 100)**8)
    kd = ko * 0.581 * np.exp(0.0365 * temp)
    moisture_content_final = ed + (moisture_content - ed) * np.exp(-2.303 * kd)
    calculated_ffmc = (59.5 * (147.2 - moisture_content_final)) / (147.2 + moisture_content_final)
    
    # ISI calculation
    f_wind = np.exp(0.047 * ws)
    f_ffmc = 91.9 * np.exp(-0.1386 * moisture_content_final) * (1.0 + (moisture_content_final**5.31) / (4.93e07))
    calculated_isi = 0.208 * f_wind * f_ffmc
    
    # DMC, DC, and FWI Approximations
    calculated_dmc = dmc_yesterday + (0.5 * temp) if temp > 0 else dmc_yesterday
    calculated_dc  = dc_yesterday + (0.1 * temp) if temp > 0 else dc_yesterday
    calculated_fwi = 0.1 * calculated_isi * calculated_dmc
    
    return {
        "FFMC": calculated_ffmc,
        "DMC": calculated_dmc,
        "DC": calculated_dc,
        "ISI": calculated_isi,
        "FWI": calculated_fwi
    }