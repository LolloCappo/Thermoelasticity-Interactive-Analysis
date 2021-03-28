import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# \dma\CFRP_0.5N encoding = 'cp1252' engine='python' skipfooter =1 decimal= ","

def estrai(path_base,N,name = None):
    if name == None:
        name = f'_{np.arange(1,N+1)}'
    else:
        N = len(name)
    keys = ['test','f','F','x','M*','tan_delta']
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
        data = pd.merge(data,df[keys],how='outer')
    data = data.set_index(['test'])
    return data

def plottaggio(data,ax1,ax2,label='',f_min=0,f_max=None,name = None,flag_colore = 0,label_colore = 'dietro'):
    if name == None:
        name = pd.Series(data.index.values).unique() # da rivedere
    if f_max == None:
        f_max = data.loc[name[0]]['f'][-1]
    data = data[(data['f'] >= f_min) & (data['f'] <= f_max)]
    N = len(name)
    M_media = np.zeros(data.loc[name[0]]['M*'].size)
    tan_media = np.zeros(data.loc[name[0]]['M*'].size)
    f = data.loc[name[0]]['f'].to_numpy()
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
            
        data.loc[name[i]].plot.scatter(x='f',y='M*',ax = ax1,label=label_temp,color='none',edgecolors = colore)
        data.loc[name[i]].plot.scatter(x='f',y='tan_delta',ax = ax2,label=label_temp,color='none',edgecolors = colore)
        M_media += data.loc[name[i]]['M*'].to_numpy()/(N)
        tan_media += data.loc[name[i]]['tan_delta'].to_numpy()/(N)

    ax1.plot(f,M_media,label='media',color = colore_media)   
    ax2.plot(f,tan_media,label='media',color = colore_media)
    ax1.set(title='M*')
    ax2.set(title='tan_delta')
    return (M_media,tan_media)

def main():
    name = []
    for i in range(1,6):
        name.append(f"CFRP_f1_100_5_1N_{i}")
    f_min = 0
    path_base = r'dma/provino 1a/'

    data_1N = estrai(path_base,1,name=name)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
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