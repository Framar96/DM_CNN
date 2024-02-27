from pydd.binary import *
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import pandas as pd
from scipy import interpolate


G = 6.67408e-11  # m^3 s^-2 kg^-1
C = 299792458.0  # m/s
MSUN = 1.98855e30  # kg
PC = 3.08567758149137e16  # m
HOUR = 3600 #s
DAY = 24*HOUR #s
YR = 365.25 * DAY  # s

ET_strain = '/gpfs/home2/marangio/project/pydd/src/pydd/noise_resources/et.dat'
data_ET = pd.read_csv(ET_strain, sep=' ',names=['Frequency','ASD'],header=None,  engine='python',index_col=False)
data_ET.drop(data_ET.head(3).index, inplace=True)



def name_check(m1,m2,rho,gamma,d,system,count):
    m1 *= MSUN
    m2 *= MSUN
    d *= PC
    rho *= MSUN/PC**3
    
    if (system=="vacuum"):
        id_str = f"M1_{m1/( MSUN):.4f}_M2_{m2/ MSUN:.4f}_dL_{d/ PC:.4f}_vacuum"
    elif (system=="PBH"):
        id_str = f"M1_{m_1 /MSUN}_M2_{m_2 / MSUN}_rho6_{rho/(MSUN/PC**3)}_gamma_{gamma}_dL_{d_l / PC}_PBH"
    
    return id_str

def save_strain(m_1, m_2, d_l, rho , gamma ,t_obs, system,count):
    m_1 *= MSUN
    m_2 *= MSUN
    d_l *= PC
    rho *= MSUN/PC**3
    
    if (system == "vacuum"):
        id_str = f"M1_{m_1 /( MSUN):.4f}_M2_{m_2 / MSUN:.4f}_dL_{d_l / PC:.4f}_vacuum"
        dd = make_vacuum_binary(m_1, m_2,np.array(0.0),None,d_l)
        
    else:
        if (system == "PBH"):

            id_str = f"M1_{m_1 /MSUN}_M2_{m_2 / MSUN}_rho6_{rho/(MSUN/PC**3)}_gamma_{gamma}_dL_{d_l / PC}_PBH"
            dd = make_dynamic_dress(m_1, m_2, rho, gamma,np.array(0.0),None,d_l)
    
    

    f_l, f_h =get_f_range(dd,t_obs)
    _fs = np.linspace(f_l, 100, 20000)
    h_f=amp(_fs,dd)* np.exp(1j * Psi(_fs, dd))
    h_f=np.real(h_f)
    length = len(_fs)
    
    fET, asdET = np.array(data_ET['Frequency'].astype(float)), np.array(data_ET['ASD'].astype(float))
    
    interpolated_asd = interpolate.interp1d(fET, asdET, kind='linear')
    
    estimated_asd = interpolated_asd(_fs)
    psd=estimated_asd **2
    sigma = 0.5 * np.sqrt(psd / _fs[0])
     # Assuming psd[0] is equivalent to psd.delta_f
    seed=None
    # Check if seed is specified and set the random seed
    if seed is not None:
        np.random.seed(seed)

    # Filter out the zero values in sigma
    not_zero = (sigma != 0)
    sigma_red = sigma[not_zero]

    # Generate Gaussian noise for the real part
    noise_real = np.random.normal(0, sigma_red)

    # Create an empty array to store the noise
    noise = np.zeros(len(sigma), dtype=np.float64)  # Assuming real part dtype is float64
    noise[not_zero] = noise_real
    
    noisy_signal = h_f + noise
    df=pd.DataFrame(noisy_signal)
    fname_out = f"strain_{id_str}"
    df.to_csv(fname_out+"_count"+str(count)+".csv", index=False)
    
M2=np.logspace(-4.5,-2.5,25)
rho_6=np.logspace(11,15,25)
gamma_s=np.linspace(2.1,2.5,25)


m1=1
count=0
while count<2:
    for m2 in M2:
        for rho in rho_6:
            for gamma in gamma_s:
                m_1     = m1   # [MSun]
                m_2     = m2  # [MSun] 10-2 10-4 0.001
                t_obs   =7*DAY  # Duration of the waveform [seconds]

                d_l     = 229*1e6   # Luminosity distance 60 Mp

                #------------------------------------------------

                filename=name_check(m_1,m_2,rho,gamma,d_l,"PBH")

                csv_exists_main = os.path.exists('/gpfs/home2/marangio/project/strain_'+filename+'_count'+str(count)+'.csv')

                if csv_exists_main==True:
                    continue

                save_strain(m_1 = m_1, 
                            m_2 = m_2, 
                            d_l = d_l, 
                            rho=rho*(m1)**(3/4),
                            gamma=gamma,
                            t_obs =t_obs,
                            system = "PBH",
                            count=count
                           )
                        
    count+=1        
                   
print("Finished")