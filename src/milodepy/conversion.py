from scipy.sparse import csr_matrix
from anndata2ri import scipy2ri

from functools import singledispatch
from scipy.sparse import csc_matrix

import rpy2.rinterface_lib.callbacks
# import anndata2ri
import logging

from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

from rpy2.robjects.packages import STAP


import numpy as np
import pandas as pd
import anndata

@singledispatch
def _py_to_r(object):
    with localconverter(default_converter):
        r_obj = numpy2ri.py2rpy(object)

    return r_obj

@_py_to_r.register
def _(object: np.ndarray):
    with localconverter(default_converter + numpy2ri.converter):
        r_obj = numpy2ri.py2rpy(object)

    return r_obj

@_py_to_r.register
def _(object: pd.DataFrame):
    with localconverter(default_converter + pandas2ri.converter):
        r_obj = pandas2ri.py2rpy(object)

    return r_obj

@_py_to_r.register
def _(object: pd.DataFrame):
    with localconverter(default_converter + pandas2ri.converter):
        r_obj = pandas2ri.py2rpy(object)

    return r_obj

@_py_to_r.register
def _(object: csr_matrix):
    with localconverter(scipy2ri.converter):
        r_obj = scipy2ri.py2rpy(object)

    return r_obj

@_py_to_r.register
def _(object: csc_matrix):
    with localconverter(scipy2ri.converter):
        r_obj = scipy2ri.py2rpy(object)

    return r_obj

def _r_to_py(object):

    r_string = '''
    .get_class <- function(object) class(object)
    '''
    r_pkg = STAP(r_string, "r_pkg")
    
    cur_class = r_pkg._get_class(object)[0]
    
    if cur_class == "dgCMatrix":
        with localconverter(scipy2ri.converter):
            r_obj = scipy2ri.rpy2py(object)

        return r_obj

    if cur_class == "dgRMatrix":
        with localconverter(scipy2ri.converter):
            r_obj = scipy2ri.rpy2py(object)

        return r_obj

    if cur_class == "matrix":
        with localconverter(default_converter + numpy2ri.converter):
            r_obj = numpy2ri.rpy2py(object)
            
        return r_obj 

    if cur_class == "data.frame":
        with localconverter(default_converter + pandas2ri.converter):
            r_obj = pandas2ri.rpy2py(object)
            
        return r_obj 

    if cur_class == "logical":
        with localconverter(default_converter + numpy2ri.converter):
            r_obj = numpy2ri.rpy2py(object)
            
        return r_obj 
    if cur_class == "character":
        with localconverter(default_converter + numpy2ri.converter):
            r_obj = numpy2ri.rpy2py(object)
            
        return r_obj 
    if cur_class == "integer":
        with localconverter(default_converter + numpy2ri.converter):
            r_obj = numpy2ri.rpy2py(object)
            
        return r_obj 
    if cur_class == "numeric":
        with localconverter(default_converter + numpy2ri.converter):
            r_obj = numpy2ri.rpy2py(object)
            
        return r_obj 


# rbase = importr("base")

r_string = '''
.get_class <- function(object) class(object)
.get_colnames <- function(mat) colnames(mat)
.get_rownames <- function(mat) rownames(mat)
.get_rownames_dge <- function(dge) rownames(dge$counts)
'''
r_pkg = STAP(r_string, "r_pkg")

# edger = importr("edgeR")

def _ad_to_rmat(adata, layer = "X"):

    mat = adata.X if layer == "X" else adata.layers[layer]
    
    obsnames = np.asarray(adata.obs_names)
    obsnames = _py_to_r(obsnames)

    varnames = np.asarray(adata.var_names)
    varnames = _py_to_r(varnames)

    r_string = '''
    assign_colnames <- function(mat, names) {
        colnames(mat) <- names
        return(mat)
    }
    assign_rownames <- function(mat, names) {
        rownames(mat) <- names
        return(mat)
    }
    print_dimnames <- function(mat) {
        print(head(rownames(mat)))
    }

    .get_class <- function(object) class(object)
    .get_colnames <- function(mat) colnames(mat)
    .get_rownames <- function(mat) rownames(mat)
    .get_rownames_dge <- function(dge) rownames(dge$counts)
    '''
    r_pkg = STAP(r_string, "r_pkg")

    rmat = _py_to_r(mat.T)
    rmat = r_pkg.assign_colnames(rmat, obsnames)
    rmat = r_pkg.assign_rownames(rmat, varnames)
    r_pkg.print_dimnames(rmat)
    

    return rmat

def _ad_to_dge(adata):
    try:
        edger = importr("edgeR")
    except:
        print("need to install rpy2 and edger")
    dge = edger.DGEList(
        counts = _ad_to_rmat(adata),
        samples = _py_to_r(adata.obs)
    )

    return dge