import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# \dma\CFRP_0.5N encoding = 'cp1252' engine='python' skipfooter =1 decimal= ","

def estrai(path_base,N,name = None,nrows = 20,normalizza=False,s=1):
    if name == None:
        name = f'_{np.arange(1,N+1)}'
    else:
        N = len(name)
    keys = ['test','f','F','x','M*','tan_delta','M','M\'']
    data = pd.DataFrame(columns = keys)
    names = ['Index','Ts','t','f','F','x','Phase','F0','x0','Tr','M','M\'','M*','tan_delta','C','C\'','C*']
    for i in range(N):
        df = pd.read_csv(path_base+f'{name[i]}.txt',delim_whitespace=True,names = names,skiprows=2,encoding = 'cp1252',engine='python',skipfooter =1)
        for key in keys:
            if key == 'test':
                df['test'] = name[i]
            else:
                df[key] = df[key].str.replace(',','.')
                df[key] = pd.to_numeric(df[key])
        df = df.head(nrows)
        data = pd.merge(data,df[keys],how='outer')
    data = data.set_index(['test'])
    data['tan_delta'] = data['tan_delta'].abs()
    data['csi'] = ottieni_smorazamento(data['tan_delta'])
    data['Stifness'] = data['F']/(data['x'])
    print(f"spostamento di {path_base} massimo {data['x'].max()/1000 } [mm]")
    data['M*'] =  data['M*']/1000
    data['M'] =  data['M']/1000
    data['M\''] =  data['M\'']/1000
    if normalizza:
        data['M*'] =  data['M*']/s#*(s**3/12)
        data['M'] =  data['M']/s#*(s**3/12)
        data['M\''] =  data['M\'']/s#*(s**3/12)
    return data

def plottaggio(data,ax1,ax2,label='test',x='f',y='M*',f_min=0,f_max=None,name = None,flag_colore = 0,label_colore = 'dietro',flag_title:bool = False,loss_factor:bool = False):
    if loss_factor:
        name_fact = 'tan_delta'
    else:
        name_fact = 'csi'
    if name == None:
        name = pd.Series(data.index.values).unique() # da rivedere
    if f_max == None:
        f_max = data.loc[name[0]]['f'][-1]
    data = data[(data['f'] >= f_min) & (data['f'] <= f_max)]
    N = len(name)
    f = data.loc[name[0]][x].to_numpy()

    M_media = np.zeros(data.loc[name[0]][y].size)
    tan_media = np.zeros(data.loc[name[0]][y].size)
    colore_media = '#%02x%02x%02x' % (np.random.randint(256),np.random.randint(256),np.random.randint(256))
    for i in range(N):
        if flag_colore == 0:
            colore = '#%02x%02x%02x' % (np.random.randint(256),np.random.randint(256),np.random.randint(256))
            label_temp = f'test {name[i]}{label}'
        elif flag_colore == 1:
            colore = colore_media
            label_temp = label
        elif flag_colore == 2:
            if label_colore in name[i]:
                colore = 'darkorange'
                label_temp = f'test {label_colore} {label}'
            else:
                label_temp = f'test {label}'
                colore = 'darkred'
        data.loc[name[i]].plot.scatter(x=x,y=y,ax = ax1,label=label_temp,color='none',edgecolors = colore)
        data.loc[name[i]].plot.scatter(x=x,y=name_fact,ax = ax2,label=label_temp,color='none',edgecolors = colore)


        M_media += data.loc[name[i]][y].to_numpy()/(N)
        tan_media += data.loc[name[i]][name_fact].to_numpy()/(N)
    if flag_colore == 1:
        ax1.legend([label])
        ax2.legend([label])
    elif flag_colore == 2:
        ax1.legend([f'test {label}',f'test {label_colore} {label}'])
        ax2.legend([f'test {label}',f'test {label_colore} {label}'])
    ax1.plot(f,M_media,label='media',color = colore_media)  
    ax2.plot(f,tan_media,label='media',color = colore_media)
    ax2.yaxis.tick_right()
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)
    if flag_title:
        ax1.set(title=y)
        ax2.set(title=name_fact)
    return (M_media,tan_media)

def dispersione(data,ax,f_min = 0,f_max = None,label='',flag=True):
    names = pd.Series(data.index.values).unique() # da rivedere
    N = len(names)
    if f_max == None:
        f_max = data.loc[names[0]]['f'][-1]

    std = 2*data[['f','M*']].groupby(['f']).std() # non campionaria a 
    M_mean = data['M*'].mean()
    ax.plot(data.loc[names[0]]['f'],std['M*']/M_mean)


    ax.set(title='(2*std)/mean M* per '+label)


def ottieni_smorazamento(loss_factor):
    delta = (1-np.sqrt(1-loss_factor**2))*2*np.pi/loss_factor
    csi = delta/np.sqrt((2*np.pi)**2 + delta**2)
    return csi

def media(data,f_lim):
    csi_mean = data[data['f']>f_lim]['csi'].mean()
    M_mean = data[data['f']>f_lim]['M*'].mean()    
    return csi_mean,M_mean

def std(data,f_lim):
    csi_std = data[data['f']>f_lim]['csi'].std()
    M_std = data[data['f']>f_lim]['M*'].std()    
    return csi_std,M_std

def main():
    name = []
    for i in range(1,6):
        name.append(f"CFRP_f1_100_5_1N_{i}")
    f_min = 0
    path_base = r'dma/provino 1a/'

    data_1N = estrai(path_base,1,name=name)
    _,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
    plottaggio(data_1N,ax1,ax2,f_min = f_min,flag_colore = 0)
    print(data_1N)
    l = 69.73
    w = 12.41
    t = 0.95
    g = l**3/(4*w*t**3)*10**3
    print(g)
    print(data_1N['F']/data_1N['x'])
    #g = 7.97*10**6
    #print(g)
    ax1.scatter(data_1N['f'],data_1N['F']/data_1N['x']*g,c='red')
    plt.show()

if __name__ == '__main__':
    main()