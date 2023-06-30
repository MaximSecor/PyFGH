import cclib
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
import os

sns.set_color_codes()

idx = 0
molecule = 10

class IRSpec(object):

    def __init__(self,logfile):
        self.data  = cclib.io.ccread(logfile)
        self.scale = 0.962 #Scaling that worked well for aromatic systems with intramolecular PCET processes
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
     

class Molecule(object):
    def __init__(self,prefix,alt_name=None):
        self.prefix  = str(prefix)
        path = os. getcwd() # Change to being CD

        for state in ['neutral','E0PT','E1PT']:                          # make these an input
            logfile = path+'/'+state+'.log'
            if os.path.isfile(logfile):
                setattr(self,state.lower(),IRSpec(path+'/opt_freq/'+state+'.log'))
                if alt_name:
                    NAME = alt_name
                else:
                    NAME = self.prefix
                if state == 'neutral':
                    setattr(getattr(self,state.lower()),'name','['+NAME+']')
                else:
                    setattr(getattr(self,state.lower()),'name','['+NAME+']$^{+}$')
  
        self.freq  = self.neutral.x # assumes neutral always exists! 

    def plotSpectra(self,curves=[],interpolate=False,num=5,xlim=None,save=None,show=True,colors=None):
        
        curves = [curves] if isinstance(curves, IRSpec) else curves

        if not xlim:
            xlim = [max(self.freq),min(self.freq)]
        
        fig,ax = plt.subplots(figsize=(8,6))
        ax.tick_params(width=2)
        
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        if interpolate:
            start = curves[0]
            finish = curves[-1]
            #ax.set_prop_cycle('color',plt.cm.Reds(np.linspace(0.3,1,num)))
            for step in np.linspace(0,num,num=num):
                curve = ((num - step)/num)*start.curve + (step/num)*finish.curve
                if step == 0:
                    label = start.name
                    color = 'black'
                    zorder = 2
                    linewidth = 2
                elif step == np.linspace(0,num,num=num)[-1]:
                    label = finish.name
                    if not colors:
                        color = 'r'
                    else:
                        color = colors
                    zorder = 3
                    linewidth = 2
                else:
                    label = None
                    color = 'gray'
                    zorder = 1
                    linewidth = 1
                plt.plot(self.freq,curve,label=label,color=color,zorder=zorder,lw=linewidth)
            # fit each plot appropriately given the curve heights within the bounds of xlim
            my_idx = (self.freq > min(xlim))*(self.freq < max(xlim))
            plt.ylim([-50,1.2*max(max(start.curve[my_idx]),max(finish.curve[my_idx]))])

        # e.g. no interpolation! finish = None
        else:
            if colors: assert len(colors) == len(curves)
            #ax.set_prop_cycle('color',plt.cm.tab10(np.linspace(0.2,0.8,len(curves))))
            #ax.set_prop_cycle('color',plt.cm.tab10(range(10)))
            for idx,spec in enumerate(curves):
                if colors:
                    color = colors[idx]
                else: color = None
                plt.plot(self.freq,spec.curve,label=spec.name,color=color)
                plt.ylim([-50,1300])
    
        plt.legend(fontsize=20)
        plt.xlim(xlim)
        plt.xticks(np.arange(min(xlim),max(xlim),75),fontsize=20)
        ax.set_xticks(np.arange(min(xlim),max(xlim),25),minor=True)
       
        plt.yticks([0],fontsize=20)
        #plt.grid(color='gray',alpha=0.4)
        #plt.grid(color='gray',alpha=0.4,which='both',ls='--')
        ax.xaxis.grid(True, which='both',ls='--',alpha=0.4,color='gray') 
        plt.xlabel(r'$\tilde{\nu}$ (cm$^{-1}$)',fontsize=24)
        plt.ylabel('Intensity (arb. units)',fontsize=24)

        if save:
            if isinstance(save,str):
                plt.tight_layout()
                plt.savefig(save,dpi=400)
            else:
                raise ValueError("'save' needs to be your filename!")
        if show:
            plt.show()
        plt.close()

    def plotDifference(self,mol1,mol2,xlim=None,save=None,show=True):
        if not xlim:
            xlim = [max(mol1.x),min(mol1.x)]
        fig,ax = plt.subplots()
        label = "$\Delta$ Abs. "+str(mol1.name)+'$-$'+str(mol2.name)
        plt.plot(self.freq,mol1.curve-mol2.curve,label=label)
        plt.legend(fontsize=20)
        plt.xlim(xlim)
        plt.xticks(np.arange(min(xlim),max(xlim),500))
        ax.set_xticks(np.arange(min(xlim),max(xlim),100),minor=True)
        #plt.ylim([-50,max(max(start.curve),max(finish.curve))])
        plt.yticks([0])
        plt.grid(color='gray',alpha=0.4)
        plt.grid(color='gray',alpha=0.4,which='minor',ls='--')
        plt.xlabel(r'$\tilde{\nu}$ (cm$^{-1}$)')
        plt.ylabel('$\Delta$ Intensity (arb. units)')
        if save:
            if isinstance(save,str):
                plt.tight_layout()
                plt.savefig(save)
            else:
                raise ValueError("'save' needs to be your filename!")
        if show:
            plt.show()
        plt.close()

    def dumpVibs(self,which,thresh=75,output=None):
        """ Return some number peaks from the frequency calculation
            Variables:
            thresh = float. minimum intensity.

            Returns: Energy (cm-1), Intensity 
        """

        idx = np.where(which.irs > thresh)
        peaks = np.hstack((which.freqs[idx].reshape(-1,1),which.irs[idx].reshape(-1,1)))[::-1]
        if output:
            np.savetxt(output,peaks,fmt='%.1f',delimiter=',')

if __name__ == '__main__':
    mol = Molecule(molecule,alt_name = "MeBIP")
    mol.plotSpectra(curves=[mol.neutral,mol.e1pt],interpolate=True,num=5,xlim=[1700,1450],show=True,save='-IRSEC_E1PT.png',colors='blue')

color_dict = {"blue": "#4472C4", "green": "#70AD47", "orange": "#ED7D31", "purple": "#7030A0", "yellow": "#BF9000", "red": "#377F80", "Orange": "#ED7D31", "Pink": "#FFOOFF", }

