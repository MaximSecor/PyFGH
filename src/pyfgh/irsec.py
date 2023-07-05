"""
===================================
===================================
              ______  _____  _   _  
              |  ___||  __ \| | | | 
 _ __   _   _ | |_   | |  \/| |_| | 
| '_ \ | | | ||  _|  | | __ |  _  | 
| |_) || |_| || |    | |_\ \| | | | 
| .__/  \__, |\_|     \____/\_| |_/ 
| |      __/ |                      
|_|     |___/                       
===================================
===================================

This module create IRSEC plots from Gaussian log files or using PySCF and the FGH functionality.

-----::::Classes included::::-----
freq_data: This class collects the frequency data.
IRSEC: This class creates elegant IRSEC plots.

"""

import cclib
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class freq_data(object):

    """
    freq_data: This class provides a brief description of the purpose and functionality of the class.

    ---Methods---
        __init__   : Initializes freq_data with data from a logfile, a IR scaling factor, the width of the lorentzians used for the IRSEC, and the IRSEC plot limits. It also initializes the frequencies and the intensities. It also creates the curve to be plotted.
        lorentzian : Creates a lorentzian to be used in creating the IR lineshape.
        curvefit   : Creates an IR lineshape to be used in plotting the IRSEC.
    
    ---Usage---
        Instantiate the class : `instance = freq_data(path_neutral, scaling_factor)`
        Plot the IR spectra   : `plt.plot(instance.curve)`
    """

    def __init__(self, logfile, scale):

        """
        __init__ method : This method initializes the class instance

        ---Args---
            logfile : str   : Path to Gaussian logfile
            scale   : float : scaling factor of IR frequencies to match experiment

        ---Params---
            data  : ccData_optdone_bool : Frequencies data
            freqs : 1D ndarray          : IR frequencies
            irs   : 1D ndarray          : IR intensities
            FWHM  : 1D ndarray          : Lorentizian width
            xlim  : float               : Curve limits
        """

        self.data  = cclib.io.ccread(logfile)
        self.scale = scale
        self.freqs = self.data.vibfreqs*self.scale
        self.irs   = self.data.vibirs
        self.FWHM  = 6
        self.xlim  = (0,4000,0.5)
        self.curvefit()
    
    def lorentzian(self,x0,h,x,gm):
        """
        lorentzian method : This method creates a lorentzian to be used in creating the IR lineshape.

        ---Args---
            x0 : str   : The center of the lorentzian
            h  : str   : The height of the lorentzian
            x  : str   : The domain of the plotted lorentzian
            gm : str   : The width of the lorentzian
        ---Returns---
            output : 1D ndarray : Lorentzian about the frequency
        """
        return (h*np.power(gm/2,2))/(np.power(x-x0,2) + np.power(gm/2,2))
        
    def curvefit(self):

        """
        curvefit method : This method creates the IR lineshape from lorentzians.

        ---Params---
            bands : zip        : Paired frequencies and associated intensities
            x     : 1D ndarray : The points on the domain to be plotted
            curve : 1D ndarray : The IR spectra curve being generated
        """

        bands = zip(self.freqs,self.irs)
        self.x = np.arange(self.xlim[0],self.xlim[1],self.xlim[2])
        self.curve = np.zeros_like(self.x) 
        for band in bands:
            self.curve += self.lorentzian(band[0],band[1],self.x,self.FWHM) 

class IRSEC:

    """
    IRSEC: This class calls the freq_data class and plot the simulated IRSEC

    ---Methods---
        __init__    : Initializes the IRSEC class and uses the freq_data class to generate the IR spectra plots used for IRSEC.
        plotSpectra : Creates an elegant IRSEC plot.
    
    ---Usage---
        Instantiate the class : `instance = IRSEC(path_neutral, path_other, scaling_factor)`
        Plot the IRSEC        : `instance.plotSpectra(0,num=5,xlim=[1650,1500],show=True,save='test.png')`
    """
    
    def __init__(self, neutral_logfile, other_files, scale):

        """
        __init__ method : This method initializes the class instance

        ---Args---
            neutral_logfile : str         : Path to the Gaussian logfile containing the neutral molecule
            other_files     : list[str]   : Paths to the other Gaussian logfile containing the other species of the molecule
            scale           : float       : scaling factor of IR frequencies to match experiment
        """
        
        self.neutral = freq_data(neutral_logfile, scale)
        self.freq  = self.neutral.x
        
        self.others = []
        for file in other_files:
            self.others.append(freq_data(file, scale))
            
    def plotSpectra(self,other_file_idx,xlim,num=5,show=True,save='-IRSEC_E1PT.png'):

        """
        plotSpectra method : This method creates an elegant IRSEC plot

        ---Args---
            other_file_idx : int         : Which species other than the neutral to use in the IRSEC
            xlim           : list[float] : The bounds of the domain of the IRSEC plot
            num            : int         : Number of intermediate IR plots in IRSEC
            show           : boolean     : If true, show plot of IRSEC
            save           : str         : Path to where IRSEC plot will be saved
        """
        
        mpl.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.linewidth'] = 2
        
        fig,ax = plt.subplots(figsize=(8,6))
        ax.tick_params(width=2)
        
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        start = self.neutral
        finish = self.others[other_file_idx]
        
        for step in np.linspace(0,num,num=num):
            curve = ((num - step)/num)*start.curve + (step/num)*finish.curve
            if step == 0:
                label = 'Neutral'
                color = 'k'
                zorder = 2
                linewidth = 3
            elif step == np.linspace(0,num,num=num)[-1]:
                label = 'E0PT'
                color = 'b'
                zorder = 3
                linewidth = 3
            else:
                label = None
                color = 'gray'
                zorder = 1
                linewidth = 1
            plt.plot(self.freq,curve,label=label,color=color,zorder=zorder,lw=linewidth)
            
        my_idx = (self.freq > min(xlim))*(self.freq < max(xlim))
        plt.ylim([-50,1.2*max(max(start.curve[my_idx]),max(finish.curve[my_idx]))])

        plt.legend(fontsize=24)
        plt.xlim(xlim)
        plt.xticks(np.arange(min(xlim),max(xlim),75),fontsize=20)
        ax.set_xticks(np.arange(min(xlim),max(xlim),25),minor=True)
        plt.yticks([0],fontsize=20)
        ax.xaxis.grid(True, which='both',ls='--',alpha=0.4,color='gray') 

        if save: plt.savefig(save,dpi=600)
        if show: plt.show()
        
        plt.close()

if __name__ == "__main__":

    path_neutral = '/Users/maximsecor/Desktop/BIP/ElectroChem/Results/BIP/opt_freq/neutral.log'
    path_other = ['/Users/maximsecor/Desktop/BIP/ElectroChem/Results/BIP/opt_freq/E0PT.log','/Users/maximsecor/Desktop/BIP/ElectroChem/Results/BIP/opt_freq/E1PT.log']
    scaling_factor = 0.962

    # mol = IRSEC(path_neutral, path_other, scaling_factor)
    # mol.plotSpectra(0,num=5,xlim=[1650,1500],show=True,save='test.png')

    mol = freq_data(path_neutral, scaling_factor)
    print(mol.freqs)
    print(type(mol.freqs))
    plt.plot(mol.curve)
    plt.show()
