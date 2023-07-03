import cclib
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class freq_data(object):

    def __init__(self, logfile, scaling_factor):
        self.data  = cclib.io.ccread(logfile)
        self.scale = scaling_factor
        self.freqs = self.data.vibfreqs*self.scale
        self.irs   = self.data.vibirs
        self.FWHM  = 6
        self.xlim  = (0,4000,0.5)
        self.curvefit()
    
    def lorentzian(self,x0,h,x,gm):
         return (h*np.power(gm/2,2))/(np.power(x-x0,2) + np.power(gm/2,2))
        
    def curvefit(self):
        bands = zip(self.freqs,self.irs)
        self.x = np.arange(self.xlim[0],self.xlim[1],self.xlim[2])
        self.curve = np.zeros_like(self.x) 
        for band in bands:
            self.curve += self.lorentzian(band[0],band[1],self.x,self.FWHM) 

class IRSEC:
    
    def __init__(self, neutral_logfile, other_files, scaling_factor):
        
        self.color_dict = {"blue": "#4472C4", "green": "#70AD47", "orange": "#ED7D31", "purple": "#7030A0", "yellow": "#BF9000", "red": "#377F80", "Orange": "#ED7D31", "Pink": "#FFOOFF", "black": "#000000"}
        self.neutral = freq_data(neutral_logfile, scaling_factor)
        self.freq  = self.neutral.x
        
        self.others = []
        for file in other_files:
            self.others.append(freq_data(file, scaling_factor))
            
    def plotSpectra(self,other_file_idx,xlim,num=5,show=True,save='-IRSEC_E1PT.png',colors='blue'):
        
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

    mol = IRSEC(path_neutral, path_other, scaling_factor)
    mol.plotSpectra(0,num=5,xlim=[1650,1500],show=True,save='test.png',colors='blue')
