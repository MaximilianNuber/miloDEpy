"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = milodepy.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys

from milodepy import __version__

__author__ = "Deijkstra"
__copyright__ = "Deijkstra"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from milodepy.skeleton import fib`,
# when using this Python module as a library.

from .conversion import _py_to_r, _r_to_py, _ad_to_dge
import scanpy as sc
import pertpy as pt
import anndata
import pandas as pd
import numpy as np
import scipy
import time
from tqdm import tqdm
milo = pt.tl.Milo()

# from patsy import dmatrix
# from inmoose.edgepy import DGEList, glmLRT, topTags, glmQLFTest
import statsmodels
import statsmodels.api as sm
import pertpy_diffexp


def assign_neighbourhoods(
    adata,
    k = 25,
    reducedDim_name = "X_pca",
    prop = 0.1,
    order = 1,
    graph_knn = False,
    transformer = None,
):
    # milo = _milo.Milo()
    mdata = milo.load(adata)
    if transformer is not None:
        # from sklearn_ann.kneighbors.annoy import AnnoyTransformer
        # transformer = AnnoyTransformer(n_neighbors=k)
        sc.pp.neighbors(mdata["rna"], use_rep=reducedDim_name, transformer = transformer)
    else:
        sc.pp.neighbors(mdata["rna"], use_rep=reducedDim_name,# , transformer = transformer
                        n_neighbors = k
                    )
    


    knn_graph = mdata["rna"].obsp["connectivities"].copy()
    graph = sc._utils.get_igraph_from_adjacency(knn_graph)
    vnames = list(range(graph.vcount()))
    graph.vs["name"] = vnames

    order1_graph = graph.copy()
        
    if order == 2:
        new_edge_list = []
        for i in range(len(vnames)):
            target = graph.neighborhood(vertices=vnames[i], order = 2)
            source = [vnames[i] for _ in target]
            new_edge_list += list(zip(source, target))
        print((new_edge_list[len(new_edge_list)-10:]))
        
        graph.add_edges(new_edge_list)
        graph.simplify()
    ### exchange the connectivities for the graph adjacency matrix
    if graph_knn:
        # make sure that the first order graph and second order graph are not the same
        # the first order graph adjacency matrix and original connectivities are not the same,
        # however they have consistently resulted in the same nhoods
        #  can someone tell me if I can exchange the connectivities for the 
        #  graph adjacency matrix? Will just do that for now
        print("differences")
        print((graph.get_adjacency_sparse()!=order1_graph.get_adjacency_sparse()).sum())
        mdata["rna"].obsp["connectivities"] = graph.get_adjacency_sparse()
    
    milo.make_nhoods(mdata, prop=prop)
        
    return mdata

def _graph_spatial_fdr(
        sample_adata: anndata.AnnData,
        neighbors_key: str | None = None,
    ):
        """FDR correction weighted on inverse of connectivity of neighbourhoods.

        The distance to the k-th nearest neighbor is used as a measure of connectivity.

        Args:
            sample_adata: Sample-level AnnData.
            neighbors_key: The key in `adata.obsp` to use as KNN graph.
        """
        # use 1/connectivity as the weighting for the weighted BH adjustment from Cydar
        w = 1 / sample_adata.var["kth_distance"]
        w[np.isinf(w)] = 0

        # Computing a density-weighted q-value.
        pvalues = sample_adata.var["PValue"]
        keep_nhoods = ~pvalues.isna()  # Filtering in case of test on subset of nhoods
        o = pvalues[keep_nhoods].argsort()
        pvalues = pvalues[keep_nhoods][o]
        w = w[keep_nhoods][o]

        adjp = np.zeros(shape=len(o))
        adjp[o] = (sum(w) * pvalues / np.cumsum(w))[::-1].cummin()[::-1]
        adjp = np.array([x if x < 1 else 1 for x in adjp])

        sample_adata.var["SpatialFDR"] = np.nan
        sample_adata.var.loc[keep_nhoods, "SpatialFDR"] = adjp

def _setup_rpy2(
    ):
        """Set up rpy2 to run edgeR"""
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr

        # numpy2ri.activate()
        # pandas2ri.activate()
        edgeR = _try_import_bioc_library("edgeR")
        limma = _try_import_bioc_library("limma")
        stats = importr("stats")
        base = importr("base")

        return edgeR, limma, stats, base

def _try_import_bioc_library(
        name: str,
    ):
        """Import R packages.

        Args:
            name (str): R packages name
        """
        from rpy2.robjects.packages import PackageNotInstalledError, importr

        try:
            _r_lib = importr(name)
            return _r_lib
        except PackageNotInstalledError:
            logger.error(f"Install Bioconductor library `{name!r}` first as `BiocManager::install({name!r}).`")
            raise

from mudata import MuData
from typing import Literal
import re
from lamin_utils import logger
from formulaic import model_matrix
def da_nhoods(
    mdata: MuData,
    design: str,
    model_contrasts: str | None = None,
    subset_samples: list[str] | None = None,
    add_intercept: bool = True,
    feature_key: str | None = "rna",
    solver: Literal["edger", "batchglm"] = "edger",
    norm_method = "TMM"
):
    """Performs differential abundance testing on neighbourhoods using QLF test implementation as implemented in edgeR.

    Args:
        mdata: MuData object
        design: Formula for the test, following glm syntax from R (e.g. '~ condition').
                Terms should be columns in `milo_mdata[feature_key].obs`.
        model_contrasts: A string vector that defines the contrasts used to perform DA testing, following glm syntax from R (e.g. "conditionDisease - conditionControl").
                            If no contrast is specified (default), then the last categorical level in condition of interest is used as the test group.
        subset_samples: subset of samples (obs in `milo_mdata['milo']`) to use for the test.
        add_intercept: whether to include an intercept in the model. If False, this is equivalent to adding + 0 in the design formula. When model_contrasts is specified, this is set to False by default.
        feature_key: If input data is MuData, specify key to cell-level AnnData object.
        solver: The solver to fit the model to. One of "edger" (requires R, rpy2 and edgeR to be installed) or "batchglm"

    Returns:
        None, modifies `milo_mdata['milo']` in place, adding the results of the DA test to `.var`:
        - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
        - `PValue` stores the p-value for the QLF test before multiple testing correction
        - `SpatialFDR` stores the p-value adjusted for multiple testing to limit the false discovery rate,
            calculated with weighted Benjamini-Hochberg procedure

    Examples:
        >>> import pertpy as pt
        >>> import scanpy as sc
        >>> adata = pt.dt.bhattacherjee()
        >>> milo = pt.tl.Milo()
        >>> mdata = milo.load(adata)
        >>> sc.pp.neighbors(mdata["rna"])
        >>> milo.make_nhoods(mdata["rna"])
        >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
        >>> milo.da_nhoods(mdata, design="~label")
    """
    try:
        sample_adata = mdata["milo"]
    except KeyError:
        logger.error(
            "milo_mdata should be a MuData object with two slots:"
            " feature_key and 'milo' - please run milopy.count_nhoods() first"
        )
        raise
    adata = mdata[feature_key]

    covariates = [x.strip(" ") for x in set(re.split("\\+|\\*|\\:", design.lstrip("~ ")))]

    # Add covariates used for testing to sample_adata.var
    sample_col = sample_adata.uns["sample_col"]
    try:
        sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
    except KeyError:
        missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
        logger.warning("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))
        raise
    sample_obs = sample_obs[covariates + [sample_col]]
    sample_obs.index = sample_obs[sample_col].astype("str")

    try:
        assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
    except AssertionError:
        logger.warning(
            f"Values in mdata[{feature_key}].obs[{covariates}] cannot be unambiguously assigned to each sample"
            f" -- each sample value should match a single covariate value"
        )
        raise
    sample_adata.obs = sample_obs.loc[sample_adata.obs_names]

    # Get design dataframe
    try:
        design_df = sample_adata.obs[covariates]
    except KeyError:
        missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
        logger.error(
            'Covariates {c} are not columns in adata.uns["sample_adata"].obs'.format(c=" ".join(missing_cov))
        )
        raise
    # Get count matrix
    count_mat = sample_adata.X.T.toarray()
    # print(count_mat.shape)
    lib_size = count_mat.sum(0)

    # Filter out samples with zero counts
    keep_smp = lib_size > 0

    # Subset samples
    if subset_samples is not None:
        keep_smp = keep_smp & sample_adata.obs_names.isin(subset_samples)
        design_df = design_df[keep_smp]
        for i, e in enumerate(design_df.columns):
            if design_df.dtypes[i].name == "category":
                design_df[e] = design_df[e].cat.remove_unused_categories()

    # Filter out nhoods with zero counts (they can appear after sample filtering)
    keep_nhoods = count_mat[:, keep_smp].sum(1) > 0

    if solver == "edger":
        # Set up rpy2 to run edgeR
        edgeR, limma, stats, base = _setup_rpy2()

        # Define model matrix
        if not add_intercept or model_contrasts is not None:
            design = design + " + 0"
        model = model_matrix(design, design_df)
        # print(model.columns)

        model.columns = [''.join(list(re.split("\\[|\\.|\\]|T", x))) for x in model.columns]
        model.columns = [x.replace(":", ".") for x in model.columns]
        

        # Fit NB-GLM
        dge = edgeR.DGEList(
            counts=_py_to_r(count_mat[keep_nhoods, :][:, keep_smp]), 
            lib_size=_py_to_r(lib_size[keep_smp])
        )
        dge = edgeR.calcNormFactors(dge, method=norm_method)

        r_model = _py_to_r(model)
        
        dge = edgeR.estimateDisp(dge, r_model)
        fit = edgeR.glmQLFit(dge, r_model, robust=True)

        # Test
        n_coef = model.shape[1]
        if model_contrasts is not None:
            r_str = """
            get_model_cols <- function(design_df, design){
                m = model.matrix(object=formula(design), data=design_df)
                return(colnames(m))
            }
            """
            from rpy2.robjects.packages import STAP

            get_model_cols = STAP(r_str, "get_model_cols")
            model_mat_cols = list(model.columns)
            # print(model_mat_cols)
            
            model_df = pd.DataFrame(model)
            
            model_df.columns = model_mat_cols

            r_model_df = _py_to_r(model_df)
            try:
                
                mod_contrast = limma.makeContrasts(contrasts=_py_to_r(model_contrasts), levels=r_model_df)
            except ValueError:
                logger.error("Model contrasts must be in the form 'A-B' or 'A+B'")
                raise
            res = base.as_data_frame(
                edgeR.topTags(edgeR.glmQLFTest(fit, contrast=mod_contrast), sort_by="none", n=np.inf)
            )
        else:
            res = base.as_data_frame(edgeR.topTags(edgeR.glmQLFTest(fit, coef=n_coef), sort_by="none", n=np.inf))

        from rpy2.robjects import conversion

        res = _r_to_py(res)
        # if not isinstance(res, pd.DataFrame):
        #     res = pd.DataFrame(res)

    # Save outputs
    # print(res.head(20))
    res.index = sample_adata.var_names[keep_nhoods]  # type: ignore
    
    if any(col in sample_adata.var.columns for col in res.columns):
        sample_adata.var = sample_adata.var.drop([x for x in res.columns if x in sample_adata.var.columns], axis=1)
    sample_adata.var = pd.concat([sample_adata.var, res], axis=1)

    # Run Graph spatial FDR correction
    _graph_spatial_fdr(sample_adata, neighbors_key=adata.uns["nhood_neighbors_key"])

def filter_nhoods(mdata):
    S = mdata["rna"].obsm["nhoods"].copy()
    if (np.asarray(mdata["rna"].obsm["nhoods"].sum(1)).ravel() == 0).sum() != 0:
        print("Careful, some cells are not in any neighborhood")
    mask = (np.asarray(mdata["rna"].obsm["nhoods"].sum(1)).ravel() != 0)
    S = S[mask, :]
    U = S.copy()
    obs_names = mdata["rna"].obs_names[mask].copy()
    set_dict = {}
    
    i = 1
    while U.shape[0] > 0:
        i += 1
        n_cells_in_nhood = np.asarray(U.sum(0))[0, :]
        
        idx_max = np.where(n_cells_in_nhood == n_cells_in_nhood.max())[0][0]
        
        cell_idxs = np.where(S[:, idx_max].toarray().ravel())[0]
        
        cell_names = obs_names[cell_idxs]
        
        U = U[U[:, idx_max].toarray().ravel() == 0, :]
   
        set_dict[str(idx_max)] = cell_names
        left_nhoods = list(set_dict.keys())
        nhs_to_use = np.asarray([int(x) for x in left_nhoods])

    index_for_refined = mdata["rna"].obs[mdata["rna"].obs.nhood_ixs_refined == 1].index[nhs_to_use.astype(int)]

    mdata["rna"].obs["nhood_ixs_refined"] = 0
    mdata["rna"].obs.loc[index_for_refined, "nhood_ixs_refined"] = 1
    
    mdata["rna"].obsm["nhoods"] = mdata["rna"].obsm["nhoods"][:, nhs_to_use.astype(int)]


from patsy import dmatrix
# def _return_null_df(adata):
#     null_df = pd.DataFrame(index = adata.var_names)
#     null_df = null_df.reset_index(names = "variable")
#     null_df["log_fc"] = np.nan
#     null_df["logCPM"] = np.nan
#     null_df["F"] = np.nan
#     null_df["p_value"] = np.nan
#     null_df["adj_p_value"] = np.nan
#     # null_df = null_df.reset_index(names = ["variable"])
#     return null_df

# def _run_edger(pdata, design, group_to_compare):
#     try:
#         # build a DGEList object
#         genes = pdata.var
#         n_genes = len(genes)
#         n_samples = pdata.n_obs
#         counts = pdata.X.T
#         anno = pdata.obs
#         # build the design matrix
#         design = dmatrix(design, data=anno)
#         for i, x in enumerate(design.design_info.column_names):
#             if group_to_compare in x:
#                 coef = i
#         dge_list = DGEList(counts=counts, samples=anno, group = [1 for _ in range(n_samples)], genes=genes)
#         # estimate the dispersions
#         dge_list.estimateGLMCommonDisp(design=design)
#         fit = dge_list.glmQLFit(design=design)
#         qlf = glmQLFTest(fit)
#         res = pd.DataFrame((topTags(qlf, n = n_genes).table))
#         res = res.drop(["ENSEMBL", "SYMBOL"], axis = 1)
#         res["FDR"] = statsmodels.stats.multitest.fdrcorrection(np.array(pd.DataFrame(res).pvalue))[1]
#         return res
#     except:
#         return _return_null_df(pdata)
def _run_edger(pdata, design, contrast, model, nhood_index):
    try:
        mod = model(pdata, design)
        mod.fit()
        res = mod._test_single_contrast(mod.contrast(**contrast))
        return anndata.AnnData(
            obs = pd.DataFrame(index = pdata.var_names),
            var = pd.DataFrame(index = [str(nhood_index)]),
            layers = dict(
                pvalue = res[["p_value"]],
                fdr_genes = res[["adj_p_value"]],
                logFC = res[["log_fc"]]
            )
        )
    except:
        return anndata.AnnData(
            obs = pd.DataFrame(index = pdata.var_names),
            var = pd.DataFrame(index = [str(nhood_index)]),
            layers = dict(
                pvalue = pd.DataFrame(data = {str(nhood_index): np.nan}, index = pdata.var_names),
                fdr_genes = pd.DataFrame(data = {str(nhood_index): np.nan}, index = pdata.var_names),
                logFC = pd.DataFrame(data = {str(nhood_index): np.nan}, index = pdata.var_names)
            )
        )
    
from joblib import Memory
import tempfile
    
def get_weights(nhoods_x):
    """
    Calculates weights based on the input matrix.

    Args:
    nhoods_x: A numpy array representing the neighborhood matrix.

    Returns:
    A numpy array of weights.
    """

    # Assuming .check_nhoods_matrix checks if the input is a valid matrix
    #  and raises an error if not. This part needs to be implemented 
    #  based on the actual checks performed in the R function.
    # check_nhoods_matrix(nhoods_x)  

    intersect_mat = np.dot(nhoods_x.T, nhoods_x)
    t_connect = np.sum(intersect_mat, axis=1)
    weights = 1 / t_connect
    return weights
from pydeseq2.dds import DeseqDataSet, DefaultInference
from pydeseq2.ds import DeseqStats
def _run_pydeseq2(pdata, design, contrast, nhood_index):
    try:
        inference = DefaultInference(n_cpus=1, joblib_verbosity = 0)
        dds = DeseqDataSet(
            adata=pdata,
            design = design,
            refit_cooks=True,
            inference=inference,
            quiet = True
        )
        dds.deseq2()
        stat_res = DeseqStats(
            dds,
            contrast=contrast,
            inference=inference,
            quiet = True
        )
        stat_res.summary()
        res = stat_res.results_df.copy()
        
        return anndata.AnnData(
            obs = pd.DataFrame(index = pdata.var_names),
            var = pd.DataFrame(index = [str(nhood_index)]),
            layers = dict(
                pvalue = res[["pvalue"]],
                fdr_genes = res[["padj"]],
                logFC = res[["log2FoldChange"]]
            )
        )
    except:
        return anndata.AnnData(
            obs = pd.DataFrame(index = pdata.var_names),
            var = pd.DataFrame(index = [str(nhood_index)]),
            layers = dict(
                pvalue = pd.DataFrame(data = {str(nhood_index): np.nan}, index = pdata.var_names),
                fdr_genes = pd.DataFrame(data = {str(nhood_index): np.nan}, index = pdata.var_names),
                logFC = pd.DataFrame(data = {str(nhood_index): np.nan}, index = pdata.var_names)
            )
        )
    
import decoupler as dc
def _run_nb(pdata, design, contrast, min_count, nhood_index):
    offset = np.log(np.asarray(pdata.X.sum(1)).ravel())
    
    genes = dc.filter_by_expr(pdata, group="milo_sample_id", min_count = int(min_count), min_total_count=round(3*1.5)
                             )
    ctdata = pdata[:, genes].copy()
    
    try:
        
        mod = pt.tl.Statsmodels(ctdata, design)
        mod.fit(sm.GLM, family = sm.families.NegativeBinomial(), offset = offset)
        res = mod._test_single_contrast(mod.contrast(**contrast))
        res["p_value"] = res.p_value.astype(float)
        return anndata.AnnData(
            obs = pd.DataFrame(index = ctdata.var_names),
            var = pd.DataFrame(index = [str(nhood_index)]),
            layers = dict(
                pvalue = res[["p_value"]],
                fdr_genes = res[["adj_p_value"]],
                logFC = res[["log_fc"]]
            )
        )
    except:
        return anndata.AnnData(
            obs = pd.DataFrame(index = ctdata.var_names),
            var = pd.DataFrame(index = [str(nhood_index)]),
            layers = dict(
                pvalue = pd.DataFrame(data = {str(nhood_index): np.nan}, index = ctdata.var_names),
                fdr_genes = pd.DataFrame(data = {str(nhood_index): np.nan}, index = ctdata.var_names),
                logFC = pd.DataFrame(data = {str(nhood_index): np.nan}, index = ctdata.var_names)
            )
        )

from itertools import islice


def chunk(arr_range, arr_size):
    arr_range = iter(arr_range)
    return iter(lambda: tuple(islice(arr_range, arr_size)), ())


from itertools import repeat
from joblib import Parallel, delayed
def de_stat_neighbourhoods(
    mdata,
    sample_col = "patient",
    design = "~condition",
    covariates = ["condition"],
    contrast = None,
    # model = pertpy_diffexp.EdgeR,
    # subset_nhoods = stat_auc$Nhood[!is.na(stat_auc$auc)],
    min_count = 3,
    n_jobs = 1,
    chunk_size = 48,
    layer = "counts",
):
    print(mdata["rna"].obs[covariates[0]])
    # print("cache function")
    assert contrast is not None

    
    def get_cells_in_nhoods(mdata, nhood_ids):
        '''
        Get cells in neighbourhoods of interest '''
        in_nhoods = np.array(mdata["rna"].obsm['nhoods'][:,nhood_ids.astype('int')].sum(1))
        ad = mdata["rna"][in_nhoods.astype(bool), :].copy()
        return ad

    covs = covariates
    print(type(covs))

    nhoods = mdata["rna"].obsm["nhoods"]
    n_nhoods = nhoods.shape[1]
    # n_nhoods = 100
    # get generator with all nhoods

    func = "sum"
    # layer = "counts"
    mdata["rna"].obs["milo_sample_id"] = mdata["rna"].obs[sample_col].copy()
    all_nhoods = (get_cells_in_nhoods(mdata, nhood_ids = np.asarray([i])) for i in range(n_nhoods))
    aggregated_nhoods = (
        sc.get.aggregate(ad, by = [sample_col] + covs, func = func, layer=layer)
        for ad in all_nhoods
    )
    def _func_to_X(ad, func):
        ad.X = ad.layers[func].copy()
        return ad
    aggregated_nhoods = (
        _func_to_X(ad, func)
        for ad in aggregated_nhoods
    )

    # if contrast is None:
    # from formulaic import model_matrix
    # FormulaicContrasts()
    # mm = model_matrix("~condition", mdata["rna"].obs)
    # n_coefs = mm.shape[1]
    # contrast = np.zeros(n_coefs)
    # contrast[n_coefs-1] = 1
    # contrast = list(contrast)
    # del mm
        
    args = zip(
        aggregated_nhoods, 
        repeat(design),
        repeat(contrast),
        repeat(min_count),
        range(n_nhoods)
    )

    #dirpath = tempfile.mkdtemp()
    #memory = Memory(dirpath, verbose = 0)
    # _run_edger_cached = memory.cache(_run_edger)

    # from multiprocessing import Pool
    start = time.time()

    assert chunk_size > n_jobs

    all_chunks = chunk(args, chunk_size)

    start_ad = anndata.AnnData(
                obs = pd.DataFrame(index = mdata["rna"].var_names),
                var = pd.DataFrame(index = [str("start")]),
                layers = dict(
                    pvalue = pd.DataFrame(data = {str("start"): np.nan}, index = mdata["rna"].var_names),
                    fdr_genes = pd.DataFrame(data = {str("start"): np.nan}, index = mdata["rna"].var_names),
                    logFC = pd.DataFrame(data = {str("start"): np.nan}, index = mdata["rna"].var_names)
                )
    )

    for cur_chunk in tqdm(all_chunks):
        res_list = Parallel(n_jobs=n_jobs, backend = "loky")(delayed(_run_edger)(ad, design, contrast, min_count, nhood_index) 
                                                             for ad, design, contrast, min_count, nhood_index in tqdm(cur_chunk))
        start_ad = sc.concat([start_ad]+res_list, axis = 1)
    
    # pval_df_list = Parallel(n_jobs=n_jobs, backend = "loky")(delayed(_run_edger)(ad, design, contrast, model) 
    #                                                          for ad, design, contrast, model in tqdm(args))
    # res_list = Parallel(n_jobs=n_jobs, backend = "loky")(delayed(_run_pydeseq2)(ad, design, contrast, nhood_index) 
    #                                                          for ad, design, contrast, nhood_index in tqdm(args))
    # pval_df_list = Parallel(n_jobs=n_jobs, backend = "loky")(delayed(_run_edger_cached)(ad, design, contrast, model) 
    #                                                          for ad, design, contrast, model in tqdm(args))

    # print("start multiprocessing")
    # from multiprocessing import get_context
    # with get_context("spawn").Pool(n_jobs) as pool:
    #     pval_df_list = pool.starmap(_run_edger_cached, args)
    # gene_order = mdata["rna"].var_names
    # pval_by_nhood = pd.concat([df.set_index("variable").loc[gene_order, :][["p_value"]] for df in pval_df_list], axis = 1)
    # FDR_across_genes = pd.concat([df.set_index("variable").loc[gene_order, :][["adj_p_value"]] for df in pval_df_list], axis = 1)
    # logfc_nhoods = pd.concat([df.set_index("variable").loc[gene_order, :][["log_fc"]] for df in pval_df_list], axis = 1)

    end = time.time()
    print("edger done in")
    print(end-start)
    print("seconds")
    return start_ad
    # return pval_df_list
    # return pval_df_list

    pval_by_nhood = pd.concat([df[["p_value"]] for df in pval_df_list], axis = 1)
    FDR_across_genes = pd.concat([df[["adj_p_value"]] for df in pval_df_list], axis = 1)
    logfc_nhoods = pd.concat([df[["log_fc"]] for df in pval_df_list], axis = 1)
    
    print("Calculating FDR across nhoods")

    n_genes = pval_by_nhood.shape[0]
    print("Adjust across nhoods")
    FDR_across_nhoods = pval_by_nhood.copy()
    for i in range(n_genes):
        
        pvalues = pval_by_nhood.iloc[i, :].values
        idx_not_nan = np.isnan(pvalues)
        
        _weights = get_weights(mdata["rna"].obsm["nhoods"][:, ~idx_not_nan])
        _weights = np.asarray(_weights)
        _weights = _weights.ravel()
        # o = order_matrix(pvalues)
        pvalues = pvalues[~idx_not_nan]
        
        o = np.argsort(pvalues)
        weights = _weights[o]
        pvalues = pvalues[o]
        _weights
        # weights = weights[o]
        adjp = np.zeros(len(o))
        adjp[o] = np.flip(np.minimum.accumulate(np.flip(np.sum(weights) * pvalues / np.cumsum(weights))))
        adjp = np.minimum(adjp, 1)
        FDR_across_nhoods.iloc[i, :] = np.nan
    
        FDR_across_nhoods.iloc[i, ~idx_not_nan] = adjp

    return pval_by_nhood, logfc_nhoods, FDR_across_genes, FDR_across_nhoods



import numpy as np
import pandas as pd
from scipy.stats import trim_mean

def _calc_factor_quantile(data, lib_size, p=0.75):
    """Calculates a quantile-based scaling factor."""
    f = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        nonzero = data[:, i] > 0
        if np.sum(nonzero) > 0:  # Check if there are any non-zero elements
            f[i] = np.quantile(np.log2(data[nonzero, i] / lib_size[i]), p) # Use np.quantile
        else:
            f[i] = 0 # or np.nan, depending on how you want to handle all-zero columns

    return f

def _calc_factor_tmm(obs, ref, libsize_obs=None, libsize_ref=None, logratio_trim=0.3, sum_trim=0.05, do_weighting=True, acutoff=-1e10):
    obs = np.asarray(obs, dtype=float)  # Convert to numeric
    ref = np.asarray(ref, dtype=float)

    nO = np.sum(obs) if libsize_obs is None else libsize_obs
    nR = np.sum(ref) if libsize_ref is None else libsize_ref

    logR = np.log2((obs / nO) / (ref / nR))
    absE = (np.log2(obs / nO) + np.log2(ref / nR)) / 2
    v = (nO - obs) / nO / obs + (nR - ref) / nR / ref

    fin = np.isfinite(logR) & np.isfinite(absE) & (absE > acutoff)

    logR = logR[fin]
    absE = absE[fin]
    v = v[fin]

    if np.max(np.abs(logR)) < 1e-6:
        return 1

    n = len(logR)
    loL = int(np.floor(n * logratio_trim) + 1) # int for indexing
    hiL = int(n + 1 - loL)
    loS = int(np.floor(n * sum_trim) + 1)
    hiS = int(n + 1 - loS)

    # Use pandas.Series.rank for consistency with R's rank
    rank_logR = pd.Series(logR).rank().values
    rank_absE = pd.Series(absE).rank().values

    keep = (rank_logR >= loL) & (rank_logR <= hiL) & (rank_absE >= loS) & (rank_absE <= hiS)


    if do_weighting:
        f = np.sum(logR[keep] / v[keep]) / np.sum(1 / v[keep])
    else:
        f = np.mean(logR[keep])
    if np.isnan(f):
        f = 0
    return 2**f

def calc_norm_factors(object, lib_size=None, method="TMM", ref_column=None, logratio_trim=0.3, sum_trim=0.05, 
                      do_weighting=True, a_cutoff=-1e10, p=0.75, 
                      scale = True,
                      **kwargs):
    # ... (previous code for input checks and preprocessing) ...

    x = np.array(object)  # Convert to NumPy array
    if np.any(np.isnan(x)):
        raise ValueError("NA counts not permitted")
    nsamples = x.shape[1]  # Number of columns (samples)

    if lib_size is None:
        lib_size = np.sum(x, axis=0)  # Column sums
    else:
        if np.any(np.isnan(lib_size)):
            raise ValueError("NA lib.sizes not permitted")
        if len(lib_size) != nsamples:
            if len(lib_size) > 1:
                print("Warning: length(lib.size) doesn't match number of samples")
            lib_size = np.resize(lib_size, nsamples)  # Replicate lib_size

    method = method.lower() # Case-insensitive

    if method == "tmmwzp":
        method = "tmmwsp"
        print("tmmwzp has been renamed to tmmwsp")

    methods = ["tmm", "tmmwsp", "rle", "upperquartile", "none"]
    if method not in methods:
       raise ValueError("Invalid method specified")

    all_zero = np.sum(x > 0, axis=1) == 0  # Rows with all zeros
    if np.any(all_zero):
        x = x[~all_zero, :]  # Remove zero rows

    if x.shape[0] == 0 or nsamples == 1:
        method = "none"

    if method == "none":
        norm_factors = np.ones(nsamples) # Return 1s if method is none
        
    if method == "tmm":
        if ref_column is None:
            f75 = _calc_factor_quantile(data=x, lib_size=lib_size, p=0.75)

            if np.median(f75) < 1e-20:
                ref_column = np.argmax(np.sum(np.sqrt(x), axis=0))
            else:
                ref_column = np.argmin(np.abs(f75 - np.mean(f75)))

        f = np.full(nsamples, np.nan) # Initialize with NaN values
        for i in range(nsamples):
            f[i] = _calc_factor_tmm(obs=x[:, i], ref=x[:, ref_column], libsize_obs=lib_size[i],
                                   libsize_ref=lib_size[ref_column], logratio_trim=logratio_trim,
                                   sum_trim=sum_trim, do_weighting=do_weighting, acutoff=a_cutoff)
        # f = np.asarray(f) # Convert to numpy array
        # f = f / np.exp(np.mean(np.log(f)))
        norm_factors = f

    # ... (other methods) ...
    
    if scale:
        return scale_to_mean_1(norm_factors)
    return norm_factors