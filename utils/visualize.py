from matplotlib import pyplot as plt
import numpy as np

def draw_figure(plot, x, y,title='', xlabel='', ylabel='',xlim=None,ylim=None,xticks=None,yticks=None,grid=True,labels=None,hidden_spines=False,hidden_ticks=False,legend=True,color=None):
    plot.title(title)
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    if hidden_ticks:
        plot.tick_params(color='#B0B0B0',direction='in')
    if hidden_spines:
        ax = plot.gca()
        ax.spines['right'].set_color('None')
        ax.spines['top'].set_color('#B0B0B0')
        ax.spines['bottom'].set_color('#B0B0B0')
        ax.spines['left'].set_color('#B0B0B0')
    if grid:
        plot.grid()
    if xlim != None:
        plot.xlim(xlim)
    if ylim != None:
        plot.ylim(ylim)
    if xticks != None:
        plot.xticks(xticks[0], xticks[1])
    if yticks != None:
        plot.yticks(yticks[0], yticks[1])
    for i in range(len(x)):
        if labels != None:
            label = labels[i]
        else:
            label = str(i)
        if color == None:
            plot.plot(x[i],y[i],label=label)
        else:
            plot.plot(x[i],y[i],label=label,color=color[i])
    if legend:
        plot.legend()

def draw_csi(csi, x):
    plt.clf()
    cnt = 0
    bar = 4
    for threshold in csi.keys():
        cnt += 1
        subplot = plt.subplot(1, len(csi), cnt)
        if cnt == 1:
            ylabel = 'CSI'
            yticks = ([0,0.2,0.4,0.6,0.8],[0,0.2,0.4,0.6,0.8])
        else:
            ylabel = ''
            yticks = ([0,0.2,0.4,0.6,0.8],['','','','',''])
        draw_figure(plt, x, csi[threshold], title='Precipitation [mm/h] $\geq$ '+str(threshold),
                    xlabel='Prediction interval [min]',ylabel=ylabel,xlim=(0,len(x[0])),ylim=(0,0.8),
                    yticks=yticks,legend=False,hidden_spines=True,xticks=([i*bar for i in range(int(len(x[0])/bar-0.01)+1)],[i*bar for i in range(int(len(x[0])/bar-0.01)+1)]))

def draw_psd(psd):
    num = psd['rapsd_gt'].shape[0] # 320
    l = int(np.log2(num))
    draw_figure(plt, [np.log2(np.linspace(1, num - 1, num - 1)), np.log2(np.linspace(1, num - 1, num - 1))],
                [np.log10(psd['rapsd_gt'][1:]) * 10, np.log10(psd['rapsd_pd'][1:]) * 10],
                title='rapsd: comparision',
                xlabel='Wavelength [km]', ylabel='Power [10$\\log_{10}(\\frac {mm/h^2}{km})$]',
                xticks=([np.log2(num / (2 ** (l-i))) for i in range(l+1)], [2 ** (l-i) for i in range(l+1)]),
                labels=['rapsd_gt', 'rapsd_pd'])

if __name__ == '__main__':
    x = np.linspace(1,64)
    y = np.linspace(64,1)
    draw_figure(plt, [np.log2(x)],[y],title='test',xlabel='Wavelength [km]', ylabel='Power [10$\\log_{10}(\\frac {mm/h^2}{km})$]',xticks=([0,1,2,3,4,5,6],[64,32,16,8,4,2,1]))
    plt.savefig('test.png')