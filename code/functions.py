import numpy as np
import igraph as ig
import networkx as nx
from collections import Counter
from scipy.stats import linregress

def createSBM(N, nc, pin, pout):
    l = N // nc
    
    pref_matrix = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if i == j:
                pref_matrix[i, j] = pin
            else:
                pref_matrix[i,j] = pout
    pref_matrix = pref_matrix.tolist()
    
    block_sizes = [nc] * l
    
    G = ig.Graph().SBM(N, pref_matrix, block_sizes)
    return G

def create_edgelist(G):
    edgelist = []
    for elem in nx.generate_edgelist(G, data=False):
        elem = elem.split()
        edgelist.append((int(elem[0]), int(elem[1])))
    return edgelist

def _createSBM(N, Nc, k, kin, output='nx', seed=None):
    
    l = N // Nc
    kout = k - kin
    pin = kin / (Nc - 1)
    pout = kout / (N - Nc)

    G = nx.planted_partition_graph(l, Nc, pin, pout, seed=seed, directed=False)
    
    if output == 'edgelist':
        return create_edgelist(G)
    if output == 'ig':
        edgelist = create_edgelist(G)
        H = ig.Graph()
        H.add_vertices(N)
        H.add_edges(edgelist)
        return H

    return G

def normalizedAvgSize(comp_sizes):
    if len(comp_sizes) < 2:
        return np.NaN

    counter = Counter(comp_sizes[1:])
    
    numerator = denominator = 0
    for size, n in counter.items():
        numerator += n*size*size
        denominator += n*size
    
    return numerator/denominator

def get_comp_data(comp_sizes):
    
    Ngcc = comp_sizes[0]
    if len(comp_sizes) > 1:
        Nsec = comp_sizes[1]
    else:
        Nsec = 0
    meanS = normalizedAvgSize(comp_sizes)
        
    return Ngcc, Nsec, meanS
                
def run(N, Nc, k, qinmin, qinmax, spacing, iterations, samples, verbose=False):
    data = []
    
    if spacing == 'Lin':
        qin_values = np.linspace(qinmin, qinmax, samples)
    else:
        qin_values = np.logspace(np.log10(qinmin), np.log10(qinmax), samples)

    if verbose:
        print('->', qin_values[0], qin_values[-1])
    for j, qin in enumerate(qin_values):

        if verbose:
            print('{} {:.6f}'.format(j, qin))
            
        #kout = pout * (N-Nc)
        #kin = k - kout
        pin = k/(Nc-1) - qin
        kin = pin * (Nc-1)
        #if verbose:
        #    print(kin)        
        for it in range(iterations):
            
            G = createSBM(N, Nc, k, kin, output='ig')
            
            C = G.transitivity_undirected(mode='zero')
            Cws = G.transitivity_avglocal_undirected(mode='zero')
            r = G.assortativity_degree(directed=False)

            components = G.components(mode='weak')
            comp_sizes = sorted([len(c) for c in components], reverse=True)
            Ngcc, Nsec, meanS = get_comp_data(comp_sizes)         

            row = [pin, it, Ngcc, Nsec, meanS, C, Cws, r]
            data.append(row)
    
    df = pd.DataFrame(data, columns=['pin', 'it', 'Ngcc', 'Nsec', 'meanS', 'C', 'Cws', 'r'])
    
    return df


def powerlaw(X, a, c):
    return c*np.array(X)**a

def getLinearReg(sizes, values, mode='logXY', return_intercept=False):
    
    if mode == 'logXY':
        X = np.log(sizes)
        Y = np.log(values)
    elif mode == 'logX':
        X = np.log(sizes)
        Y = np.array(values)
    elif mode == 'logY':
        X = np.array(sizes)
        Y = np.log(values)
    else:
        X = np.array(sizes)
        Y = np.array(values)
    
    slope, intercept, r_value, p_value, slope_err = linregress(X, Y)
    r2 = r_value**2
    Y_pred = intercept + X*slope
    
    if mode in ['logY', 'logXY']:
        Y_pred = np.exp(Y_pred)
    if return_intercept:
        return Y_pred, slope, slope_err, r2, intercept
    else:
        return Y_pred, slope, slope_err, r2