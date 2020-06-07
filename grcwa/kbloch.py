import numpy as np

def Lattice_Reciprocate(L1,L2):
    '''Given two lattice vectors L1,L2 in the form of (Lx,Ly), returns the
    reciprocate vectors Lk/(2*pi)
    '''

    assert type(L1) == list and type(L2) == list, 'Lattice vectors should be in list format.'
    assert len(L1) == 2,'Both x,y components of Lattice vector L1 are required.'
    assert len(L2) == 2,'Both x,y components of Lattice vector L2 are required.'
    
    d = L1[0]*L2[1]-L1[1]*L2[0]

    Lk1 = np.array([L2[1]/d, -L2[0]/d])
    Lk2 = np.array([-L1[1]/d, L1[0]/d])

    return Lk1,Lk2

def Lattice_getG(nG,Lk1,Lk2,method=0):
    '''
    The G is defined to produce the following reciprocal vector:
    k = G[:,0] Lk1 + G[:,1] Lk2 (both k and Lk don't include the 2pi factor)
    
    method:0 for circular truncation, 1 for parallelogramic truncation
    '''
    assert type(nG) == int, 'nG must be integar'
    
    if method == 0:
        G,nG = Gsel_circular(nG, Lk1, Lk2)
    elif method == 1:
        G,nG = Gsel_parallelogramic(nG, Lk1, Lk2)
    else:
        raise Exception('Truncation scheme is not included')

    return G,nG

def Lattice_SetKs(G, kx0, ky0, Lk1, Lk2):
    '''
    Construct kx,ky including all relevant orders, given initial scalar kx,ky
    2pi factor is now included in the returned kx,ky
    '''

    kx = kx0 + 2*np.pi*(Lk1[0]*G[:,0]+Lk2[0]*G[:,1])
    ky = ky0 + 2*np.pi*(Lk1[1]*G[:,0]+Lk2[1]*G[:,1])

    return kx,ky


def Gsel_parallelogramic(nG, Lk1, Lk2):
    ''' From Liu's gsel.c'''
    u = np.linalg.norm(Lk1)
    v = np.linalg.norm(Lk2)
    uv = np.dot(Lk1,Lk2)

    NGroot = int(np.sqrt(nG))
    if np.mod(NGroot,2) == 0:
        NGroot -= 1
        
    M = NGroot//2

    xG = range(-M,NGroot-M)
    G1,G2 = np.meshgrid(xG,xG,indexing='ij')
    G1 = G1.flatten()
    G2 = G2.flatten()

    # sorting
    Gl2 = G1**2*u**2+G2**2*v**2+2*G2*G1*uv
    sort = np.argsort(Gl2)
    G1 = G1[sort]
    G2 = G2[sort]

    # final G
    nG = NGroot*NGroot    
    G = np.zeros((nG,2),dtype=int)
    G[:,0] = G1[:nG]
    G[:,1] = G2[:nG]    

    return G, nG

def Gsel_circular(nG, Lk1, Lk2):
    '''From Liu's gsel.c.
    NG * |u x v| is approximately the area in k-space we will need
    cover with a circular disc. (u and v are the 2 shortest lattice
    vectors) From the area, we can find the radius (and round it
    up). Then, we can find the minimum extends in each of the two
    lattice directions.
    '''
    u = np.linalg.norm(Lk1)
    v = np.linalg.norm(Lk2)
    uv = np.dot(Lk1,Lk2)
    uxv = Lk1[0]*Lk2[1] - Lk1[1]*Lk2[0]
    circ_area = nG * np.abs(uxv)
    circ_radius = np.sqrt(circ_area/np.pi) + u+v;

    u_extent = 1+int(circ_radius/(u*np.sqrt(1.-uv**2/(u*v)**2)))
    v_extent = 1+int(circ_radius/(v*np.sqrt(1.-uv**2/(u*v)**2)))

    uext21 = 2*u_extent+1;
    vext21 = 2*v_extent+1;

    xG = range(-u_extent,uext21-u_extent)
    yG = range(-v_extent,vext21-v_extent)
    G1,G2 = np.meshgrid(xG,yG,indexing='ij')
    G1 = G1.flatten()
    G2 = G2.flatten()

    # sorting
    Gl2 = G1**2*u**2+G2**2*v**2+2*G2*G1*uv
    sort = np.argsort(Gl2)
    G1 = G1[sort]
    G2 = G2[sort]
    Gl2 = Gl2[sort]

    nGtmp = uext21*vext21
    
    if nG < nGtmp:
        nGtmp = nG

    # removing the part outside the cycle
    tol = 1e-10*max(u**2,v**2)
    for i in range(nGtmp-1,-1,-1):
        if np.abs(Gl2[i]-Gl2[i-1])>tol:
            break;
    nG = i
    
    # final G
    G = np.zeros((nG,2),dtype=int)
    G[:,0] = G1[:nG]
    G[:,1] = G2[:nG]

    return G,nG
