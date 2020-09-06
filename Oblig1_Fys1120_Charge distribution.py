#//============================================================================================
#// Fys1120 - Oblig_1 Studying a charge distribution, Thashian Thayananthan, Sami Shafi
#//============================================================================================
#// Programmen under er den numeriske løsningen for feltet og potensialet.
#// Koden som blir brukt her kommer til å bli vist i små deler i PDF-en. 
#//============================================================================================
 
import numpy as np
import matplotlib.pyplot as plt

#//============================================================================================
#Funksjon for Feltet; r,Q,R
#//============================================================================================
def epotlist(r,Q,R):
    V = 0
    for i in range(len(R)):
        Ri = r - R[i]
        qi = Q[i]
        Rinorm = np.linalg.norm(Ri)
        V = V + qi/Rinorm
    return V

R = []; Q = []

#//============================================================================================
#Framgangsmåten for å finne ladningene rundt ringen.
#//============================================================================================
theta = np.linspace(0,np.pi*2,100)

a = 50
for i in range(1,100):
    q0 = 1.0
    r0 = np.array([np.cos(theta[i])*a,np.sin(theta[i])*a,0])
    R.append(r0)
    Q.append(q0)

#//============================================================================================
#Antal målinger som skal skje (N).
#//============================================================================================
Lx = 150; Ly = 150
N = 80
x = np.linspace(-Lx,Lx,N)
y = np.linspace(-Ly,Ly,N)
rx,ry = np.meshgrid(x,y)
V = np.zeros((N,N),float)

for i in range(len(rx.flat)):
    r = np.array([rx.flat[i],ry.flat[i],0])
    V.flat[i] = epotlist(r,Q,R)

E_y,E_x = np.gradient(-V)
Emag = np.sqrt(E_x**2 + E_y**2)

minlogEmag = min(np.log10(Emag.flat))
scaleE = np.log10(Emag) - minlogEmag
uEx = E_x / Emag
uEy = E_y / Emag

#//============================================================================================
#Plotting av koden
#//============================================================================================
levels = np.arange(0, 3.5+0.2, 0.2)
cmap = plt.cm.get_cmap('plasma')

plt.figure(figsize=(16,8))
ax1 = plt.subplot(1,2,1)
plt.contourf(rx,ry,V,10, cmap=cmap, levels=levels, extend='both')
skip = (slice(None, None, 2), slice(None, None, 2))
plt.quiver(rx,ry,uEx*scaleE,uEy*scaleE)
ax1.set_aspect('equal', 'box')

ax2 = plt.subplot(1,2,2)
plt.contourf(rx,ry,V,10, cmap=cmap, levels=levels, extend='both')
plt.streamplot(rx,ry, E_x, E_y)
ax2.set_aspect('equal', 'box')

ax1.set_title("Electric Field Using quiver")
ax2.set_title("Electric Field Using streamplot")

plt.show()

#//============================================================================================
#Denne delen av koden er hvor vi undersøker hvordan feltet og
#potensialet oppfører seg. Dette sammenlikner vi med både det presise og
#for forenklede metoden.
#//============================================================================================
xs = np.linspace(-10,10,1000)
epsilon0 = 8.854187812e-12
R = 1

Exact_field  = xs/(epsilon0 * 4 * np.pi * (xs**2+R**2)**(3/2))
Approx_field = xs/(epsilon0 * 2 * np.pi * (xs**2+R**2)**(3/2))

Exact_pot = abs(epsilon0 * 4 * np.pi * 1/(np.sqrt(xs**2+R**2)))
Approx_pot = abs(epsilon0 * 2 * np.pi * 1/(np.sqrt(xs**2+R**2)))

ax1 = plt.subplot(1,2,1)
plt.title('Plot of Electric-Field')
plt.plot(xs, Exact_field, color='#E91E63', label="E, none simplified")
plt.plot(xs,Approx_field, color='black', label="E, simplified")
plt.xlabel('x')
plt.ylabel('E')
plt.legend()
plt.grid()

ax2 = plt.subplot(1,2,2)
plt.title('Plot of Electric-potential')
plt.plot(xs, Exact_pot, color='#E91E63', label="V(r), none simplified")
plt.plot(xs, Approx_pot, color='black', label="V(r), simplified")
plt.xlabel('x')
plt.ylabel('V')
plt.legend()
plt.grid()
plt.show()
