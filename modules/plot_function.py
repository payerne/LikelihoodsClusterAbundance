import numpy as np
import matplotlib.pyplot as plt

def likelihood_load(file = None, only_one=False):
    if only_one:
        Om, y = np.load(file, allow_pickle=True)
        lnL_tot = np.array([y[i] for i in range(len(Om))])
        L = np.exp(lnL_tot-np.max(lnL_tot))
        P = L/np.trapz(L, Om)
        result = {}
        result['Om'] = Om
        result['P'] = P
    else:
        Om, y = np.load(file, allow_pickle=True)
        lnL_tot = np.array([y[i][0] for i in range(len(Om))])
        lnL_SN = np.array([y[i][1] for i in range(len(Om))])
        L_SN = np.exp((lnL_SN - np.max(lnL_SN)))
        P_SN = L_SN/np.trapz(L_SN, Om)
        L_SSCSN = np.exp(lnL_tot-np.max(lnL_tot))
        P_SSCSN = L_SSCSN/np.trapz(L_SSCSN, Om)
        result = {}
        result['Om'] = Om
        result['PSN'] = P_SN
        result['PSNSSC'] = P_SSCSN
        result['lnW'] = lnL_tot - lnL_SN
    return result

def splot(file = None, name = None, color = None, ls = None, name_save=None, range_plot_W = [0, 1]):

    lnL_tot = []
    lnL_SN = []
    Om_axis = []
    Nobs = []
    for f in file:
        Om, y = np.load(f, allow_pickle=True)
        lnL_tot.append([y[i][0] for i in range(len(Om))])
        lnL_SN.append([y[i][1] for i in range(len(Om))])
        Om_axis.append(Om)

    fig, ax = plt.subplots(1, 2, figsize = (10, 4), sharex = True)
    
    ax[0].set_ylim(0., 120)
    ax[1].set_ylim(range_plot_W[0], range_plot_W[1])
    ax[0].set_xlim(0.27, 0.34)
    ax[0].vlines(0.30711, -3, 1020, zorder=0, color='k', lw=1, )
    ax[1].vlines(0.30711, -3, 1020, zorder=0, color='k', lw=1, )
    ax[0].set_ylabel(r'P($\Omega_m$|data)', fontsize=13)
    ax[1].set_ylabel(r'$\ln\langle W \rangle_{\rm SSC}$', fontsize=13)
    x = Om_axis
    for i, lnL_SN_i in enumerate(lnL_SN):

        mask = (x[i] > 0.1)*(x[i] < 0.4)
        L_SN1 = np.exp((lnL_SN_i - np.mean(lnL_SN_i)))
        #print( )
        P_SN2 = L_SN1[mask]/np.trapz(L_SN1[mask], x[i][mask])
        xmean = np.trapz(P_SN2*x[i][mask],x[i][mask])
        sigma2SN = np.trapz(P_SN2*(x[i][mask] - xmean)**2,x[i][mask])
        print(f'{xmean:.5f} {sigma2SN**.5:.5f}')
        ax[0].plot(x[i][mask], P_SN2, ls='-' ,c=color[i], alpha=1, lw=1, label = name[i])
        L_SSCSN1 = np.exp(lnL_tot[i]-np.max(lnL_tot[i]))
        P_SSCSN2 = L_SSCSN1[mask]/np.trapz(L_SSCSN1[mask], x[i][mask])
        xmean = np.trapz(P_SSCSN2*x[i][mask],x[i][mask])
        sigma2SSC = np.trapz(P_SSCSN2*(x[i][mask] - xmean)**2,x[i][mask])
        print(f'{xmean:.5f} {sigma2SSC**.5:.5f}')
        print('ratio = ' + str(((sigma2SN**.5)/(sigma2SSC**.5))**(-1)))
        ax[0].plot(x[i][mask], P_SSCSN2, ls='--' ,c=color[i], zorder=3, alpha=1, lw=1)
        ax[1].plot(x[i][mask],(np.array(lnL_tot[i])[mask]-np.array(lnL_SN[i])[mask]), color = color[i], ls = ls[i], lw=1)

    for i in range(2):
        ax[i].tick_params(axis='both', which = 'major', labelsize= 12)
        ax[0].legend(fontsize=10)
        ax[i].set_xlabel(r'$\Omega_m$', fontsize=13)
        ax[0].plot([],[],'--k',label = 'HLC/GPC')
    plt.savefig(name_save, bbox_inches='tight', dpi=300)

    return 0

def splot_standard(file = None, name = None, color = None, ls = None, name_save=None):

    lnL_tot = []
    Om_axis = []
    Nobs = []
    for f in file:
        Om, y = np.load(f, allow_pickle=True)
        lnL_tot.append([y[i][1] for i in range(len(Om))])
        Om_axis.append(Om)
        
    plt.figure(figsize = (7, 4),)
    
    plt.ylim(0, 120)
    plt.xlim(0.275, 0.35)
    plt.vlines(0.30711, -3, 1020, zorder=0, color='k', lw=1, )
    plt.ylabel(r'P($\Omega_m$|data)', fontsize=13)
    plt.ylabel(r'$\ln\langle W \rangle$', fontsize=13)
    x = Om_axis
    for i, lnL_SN_i in enumerate(lnL_tot):

        mask = (x[i] > 0.1)*(x[i] < 0.4)
        L_SN1 = np.exp((lnL_SN_i - np.mean(lnL_SN_i)))
        #print( )
        P_SN2 = L_SN1[mask]/np.trapz(L_SN1[mask], x[i][mask])
        xmean = np.trapz(P_SN2*x[i][mask],x[i][mask])
        sigma2SN = np.trapz(P_SN2*(x[i][mask] - xmean)**2,x[i][mask])
        print(f'{xmean:.5f} {sigma2SN**.5:.5f}')
        plt.plot(x[i][mask], P_SN2,c=color[i], alpha=1, lw=1, label = name[i], ls = ls[i])

    plt.tick_params(axis='both', which = 'major', labelsize= 12)
    plt.legend()
    plt.xlabel(r'$\Omega_m$', fontsize=13)
    #plt.savefig(name_save, bbox_inches='tight', dpi=300)

    return 0
