import os
import sys
import argparse
import numpy as np
import pandas as pd
from functions import createSBM, get_comp_data


def run(N, Nc, k, Xmin, Xmax, spacing, iterations, samples, verbose=False):
    data = []
    
    if spacing == 'Lin':
        X = np.linspace(Xmin, Xmax, samples)
    else:
        X = np.logspace(np.log10(Xmin), np.log10(Xmax), samples)

    if verbose:
        print('->', X[0], X[-1])
    for j, x in enumerate(X):

        qin = x/Nc**2

        if verbose:
            print('{} {:.6f}'.format(j, qin))
            
        pin = k/(Nc-1) - qin
        kin = pin * (Nc-1)
        kout = k - kin
        pout = kout / (N-Nc)

        if pin < 0 or pin > 1:
            continue

        #print(qin, pin, pout)
       
        for it in range(iterations):
            
            G = createSBM(N, Nc, pin, pout)
            
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

def parse_args():

    parser = argparse.ArgumentParser(description='Run different SBM with constant <k> and different (pin, pout)')
    parser.add_argument('--N', required=True, type=int, help='Network size')
    parser.add_argument('--Nc', required=True, type=int, help='Block size')
    #parser.add_argument('--l', type=int, help='Number of blocks')
    parser.add_argument('--meank', required=True, type=float, help='Average degree')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples to be taken')
    parser.add_argument('--it', type=int, default=10, help='Iterations per sample')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existent file')
    parser.add_argument('--verbose', action='store_true', help='Overwrite existent file')
    parser.add_argument('--data_dir', type=str, default='../data', help='Output directory')
    parser.add_argument('--xmin', required=True, type=float, help='Lower value in X axis')
    parser.add_argument('--xmax', required=True, type=float, help='Higher value in X axis')
    parser.add_argument('--spacing', type=str, default='Log', help='Linear or logarithmic spacing')

    args = parser.parse_args()

    return args

## Command line parameters
args = parse_args()
N = args.N
Nc = args.Nc
k = args.meank
xmin = args.xmin
xmax = args.xmax
iterations = args.it
samples = args.samples
spacing = args.spacing
data_dir = args.data_dir

csv_file_name = 'N{}_Nc{}_k{:.2f}_xmin{:.4f}_xmax{:.4f}_samples{}_it{}_{}.csv'.format(
    N, Nc, k, xmin, xmax, samples, iterations, spacing
)
full_csv_file_name = os.path.join(data_dir, csv_file_name)

if os.path.isfile(full_csv_file_name) and not args.overwrite:
    df = pd.read_csv(full_csv_file_name)
else:
    df = run(N, Nc, k, xmin, xmax, spacing, iterations, samples, verbose=args.verbose)
    df.to_csv(full_csv_file_name)