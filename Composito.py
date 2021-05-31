import numpy as np 
#   sk [mm]   Ek0 [Mpa]   Ek90[Mpa]   v    G [Mpa]  
rc_200 = {'s':0.26,'Ex':53522,'Ey':53522,'v':0.03,'G':2929,'fabric':True,'theta':0}
rc_400 = {'s':0.49,'Ex':53522,'Ey':53522,'v':0.03,'G':2929,'fabric':True,'theta':0}
xc_400 = {'s':0.47,'Ex':57770,'Ey':57770,'v':0.03,'G':3186,'fabric':True,'theta':0}
uc_300 = {'s':0.33,'Ex':118454,'Ey':7104,'v':0.29,'G':3531,'fabric':False,'theta':0}
resina = {'s':0.1,'Ex':0,'Ey':0,'v':0,'G':0,'fabric':False}
laminato = [rc_200,rc_400,xc_400,uc_300,uc_300,uc_300,rc_400,uc_300,uc_300,uc_300,xc_400,rc_400]

s = 0
for i in laminato:
    E_l = i['Ex']*i['s']
    E_t = i['Ey']*i['s']
    s += i['s']
E_l = E_l/s
print(E_l)
print(E_t)


def Q_ply(ply,th=0, fabric = False):
    ''' S=[Q_k]*e
    '''
    if th == 0:
        th = ply['theta']
    E_l = ply['Ex']
    E_l = ply['Ey']
    G = ply['G']
    v_lt = ply['v']
    v_tl = v_lt*(E_l/E_t)
    if fabric:
        Q = np.array([[(E_l+E_t)/2/(1-v_lt*v_tl),v_lt*E_t/(1-v_lt*v_tl),0],[v_lt*E_t/(1-v_lt*v_tl),(E_l+E_t)/2/(1-v_lt*v_tl),0],[0,0,G]])
    else:
        Q = np.array([[E_l/(1-v_lt*v_tl),v_lt*E_t/(1-v_lt*v_tl),0],[v_lt*E_t/(1-v_lt*v_tl),E_t/(1-v_lt*v_tl),0],[0,0,G]])
    
    n = np.cos(th*(np.pi/180))
    m = np.sin(th*(np.pi/180))
    Te = np.array([[m**2,n**2,n*m],[n**2,m**2,-n*m],[-2*n*m,2*n*m,m**2-n**2]])
    Ts1 = np.array([[m**2,n**2,-2*n*m],[n**2,m**2,2*n*m],[n*m,-n*m,m**2-n**2]])
    S = np.dot(Ts1,np.dot(Q,Te))
    return  Q

def S_ply(ply,th=0):
    ''' e=[S_k]*S
    '''
    E_l = ply['Ex']
    E_l = ply['Ey']
    G = ply['G']
    v_lt = ply['v']
    v_tl = v_lt*(E_l/E_t)
    S = np.array([[1/E_l,-v_tl/E_t,0],[-v_lt/E_l,1/E_t,0],[0,0,1/G]])

    n = np.cos(th*(np.pi/180))
    m = np.sin(th*(np.pi/180))
    Te1 = np.array([[m**2,n**2,-n*m],[n**2,m**2,n*m],[2*n*m,-2*n*m,m**2-n**2]])
    Ts = np.array([[m**2,n**2,2*n*m],[n**2,m**2,-2*n*m],[-n*m,n*m,m**2-n**2]])
    S = np.dot(Te1,np.dot(S,Ts))
    return  S

def laminate(laminato):
    '''
    input:
    - laminato : dict {'s': [mm],'Ex':[Mpa],'Ey':[Mpa],'G':[Mpa],'v':[add],'fabric':[bool]}
    output:
    - A,B,D
    '''
    s = 0
    for lamina in laminato: # definisco lo Spessore
        s += lamina['s']
    z_0 = s/2 # Cordinata piano strato
    A = np.zeros([3,3])
    B = np.zeros([3,3])
    D = np.zeros([3,3])
    theta = 0
    for lamina in laminato:
        s_k = lamina['s']
        theta = lamina['theta']
        z_k = z_0 - s_k/2 # centro
        z_0 += -s_k # limite inferiore
        # print(f'centro {z_k}, Sk {z_k+s_k/2} sk-1 {z_0}')
        Q_k = Q_ply(lamina,theta,fabric = lamina['fabric'])
        A += Q_k*s_k
        B += Q_k*(s_k*z_k/2)
        D += Q_k*((z_k+ s_k/2)**3-(z_0)**3)/3
    return A,B,D
#
print('-------------------------------')

sigma = [100,0,0]
th = np.arange(0,90)
res_e = np.zeros([th.size,3])
k = 0
composito = {'Ex':117.5*10**3,'Ey':9.8*10**3,'G':9.8*10**3,'v':0.3,'theta':0}
for theta in th:
    S_k = S_ply(composito,theta)
    res_e[k,:] = np.dot(S_k,sigma)
    k += 1

import matplotlib.pyplot as plt
_,ax = plt.subplots()
ax.plot(th,res_e[:,0],label = 'ex')
ax.plot(th,res_e[:,1],'tab:red',label = 'ey')
ax.plot(th,2*res_e[:,2],'tab:grey',label = 'exy')
ax.legend()
plt.show()
#
print('--------------------------------')
laminato = [rc_400,uc_300,uc_300,uc_300,resina,xc_400]
test = {'s': 0.6,'Ex':117.5*10**3,'Ey':9.8*10**3,'G':9.8*10**3,'v':0.3,'fabric': False,'theta':0}
laminato = [test,test,test]

A,B,D = laminate(laminato)
print('- A')
print(A)
print('- B')
print(B)
print('- D')
print(D)
t = s
print(D[0,0]/(t**3/12))
