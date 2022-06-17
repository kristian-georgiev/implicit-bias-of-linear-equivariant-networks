import os
import glob
import numpy as np
from matplotlib import pyplot as plt

@parallel(ncpus=40, timeout=20)
def parse_val(e):
    """
    Input: a string containing GAP expression of Cyclotomics, i.e. "E(5)^2 - E(3)"
    Output: a Sage object of Complex type
    
    This is the only point where Sage is used. If you have trouble running Sage,
    its use can be replaced by a custom function to evaluate GAP Cyclotomics.
    """
    z = libgap.eval(e).sage()
    if z == conjugate(z):
        return real(z)
    return complex(z)

    
def parse_mx(filename):
    """
    Input: a file containing some output of fourierbasisdecomp.g for a given group
    Output: a three-tuple containing the dimensions of the irreps,
            an index of the group elements (since there is no canonical ordering of a group!),
            and the Fourier basis matrix of the group.
    """
    mx = []
    rownum = 0

    with open(filename) as f:
        s = f.readlines()[0]
        for (rownum,x) in enumerate(s.split('],')):

            x = x.replace('[', '')
            x = x.replace(']', '')
            x = x.replace(' ', '')
            if rownum == 0:
                index_group = x.split(',')
                continue
            
            if rownum == 1:
                irrep_dims = list([int(d) for d in x.split(',')])
                continue           
            row = [x[-1] for x in list(parse_val([e for e in x.split(',')]))]
            mx.append(row)
    
    mx = np.matrix(mx, np.complex64)
#     window_begin = 0
#     for d in irrep_dims:
        
#         #note that the GAP bases are *unscaled* and we do it here for precision reasons
#         row_scaling_factor = np.sqrt(d / len(index_group))
        
#         for _ in range(d):
#             window_end = window_begin + d
#             mx[window_begin:window_end] *= row_scaling_factor
#             window_begin = window_end

    return irrep_dims, index_group, mx

def parse_cayley(filename):
    """
    Input: a file containing some output of get_cayleys.g for a given group
    Output: a Cayley table of the group elements renamed by index
    """
    with open(filename) as f:
        c = f.readlines()[0].split(', ')[:-1]
    n = int(np.sqrt(len(c)))
    c_np = np.array(c).reshape(n, n)
    first_row = c_np[0]
    str_to_ind = {first_row[ind]: ind for ind in range(n)}
    cayley = np.zeros_like(c_np)
    for i in range(n):
        for j in range(n):
            cayley[i][j] = str_to_ind[c_np[i][j]]

    return cayley.astype(np.int32)

# Manual script to convert some precomputed group data into a more convenient Numpy binary format
# This is not intended to be elegant.
for file in glob.glob('./data/cayley/*'):
    if '.npy' not in file and file + '.npy' not in glob.glob('./cayley/*'):
        print(file)
        cayley = parse_cayley(file)
        np.save(file + '.npy', cayley)
        
for file in glob.glob('./data/unscaled_bases/*'):
    if '.npy' not in file and not os.path.isfile(file + '.npy'):
        irrep_dims, index_group, base = parse_mx(file)
        print(base.shape)
        np.save(file + '.npy', base)

# Example: the unitary Fourier basis matrix for the dihedral group with 60 elements
# m = parse_mx('data/unscaled_bases/D60')[2]
# print(m)
# x = (m.getH() * m)
# print('\n')
# plt.imshow(x.real); plt.show(); plt.imshow(x.imag); plt.colorbar(); plt.show()

# Example: the Fourier basis matrix and Cayley table for the group (C5 x C5) : Q8.
# This group was chosen because it is easily accessible in GAP as SmallGroup(200,44)
# and it has irreps of dimension 8 that are compatible with the RepnDecomp libary.
# irrep_dims, index_group, dft_matrix = parse_mx('data/unscaled_bases/CCQ200')

# print(irrep_dims)
# plt.imshow(dft_matrix.real); plt.show()

# cayley = parse_cayley('data/cayley/CCQ200_Cayley')
# plt.imshow(cayley); plt.show()
