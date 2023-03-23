#!/usr/bin/env python

from .utils import parsing,regression_utils,sagpr_utils
import scipy.linalg
import sys
import numpy as np
from ase.io import read

###############################################################################################################################

def predict(rank, weights, kernel, threshold=1e-8, asymmetric=''):

    # This is a wrapper that calls python scripts to do SA-GPR with pre-built L-SOAP kernels.
    for k in range(len(kernel)):
        ns, nt = kernel[k].shape[:2]
    # Initialize variables describing how we get a full tensor from its spherical components
    if (asymmetric == ''):
        [sph_comp,degen,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list] = sagpr_utils.get_spherical_tensor_components(['1.0 ' * 3**rank for i in range(nt)],rank,threshold)
    else:
        if (len(asymmetric) != 2):
            print("ERROR: with asymmetric option, two arguments must be given!")
            sys.exit(0)
        if (rank <= 1):
            print("ERROR: scalar properties cannot be asymmetric!")
            sys.exit(0)
        elif (rank == 2):
            tens = ' '.join(np.concatenate(np.array(read(asymmetric[0],':')[0].info[asymmetric[1]]).astype(str)))
        else:
            tens = ' '.join(np.array(read(asymmetric[0],':')[0].info[asymmetric[1]]).astype(str))
        [sph_comp,degen,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list] = sagpr_utils.get_spherical_tensor_components([tens for i in range(nt)],rank,threshold)

    outvec = []
    for l in range(len(degen)):
        # Find a prediction for each spherical component
        lval = keep_list[l][-1]
        str_rank = ''.join(map(str,keep_list[l][1:]))
        if (str_rank == ''):
            str_rank = ''.join(map(str,keep_list[l]))
        outvec.append(sagpr_utils.do_prediction_spherical(kernel[l],rank_str=str_rank,weight_array=weights[l], weightfile='', outfile=''))

    ns = int(len(outvec[0]) / degen[0])
    predcart = regression_utils.convert_spherical_to_cartesian(outvec,degen,ns,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list)

    return predcart

