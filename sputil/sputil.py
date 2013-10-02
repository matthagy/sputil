'''Utilities for working with scipy.sparse matrices
'''

import numpy as np
import scipy.sparse as sp

__all__ = ['hstack_csc_cols', 'vstack_csr_rows',
           'hstack_csc_mats', 'vstack_csr_mats',
           'vstack_csc_cols', 'vstack_csc_mats',
           'get_select_csc_cols', 'get_select_csr_rows',
           'get_csc_cols', 'get_csr_rows',
           'remove_csr_rows']



class unsafe_cs_matrix(object):
    '''Removes all sanity checks in compressed matrix creation
    '''

    def __init__(self, state, shape):
        self.data, self.indices, self.indptr = state
        self._shape = shape

class unsafe_csc_matrix(unsafe_cs_matrix, sp.csc_matrix):
    pass

class unsafe_csr_matrix(unsafe_cs_matrix, sp.csr_matrix):
    pass


def statck_sparse_1d(mats):
    acc_data = []
    acc_indptr = []
    acc_indices = []
    sum_indices = 0
    for mat in mats:
        acc_data.append(mat.data)
        acc_indptr.append(sum_indices)
        acc_indices.append(mat.indices)
        sum_indices += len(mat.indices)
    acc_indptr.append(sum_indices)

    data = np.concatenate(acc_data)
    indptr = np.array(acc_indptr, dtype=np.intc)
    indices = np.concatenate(acc_indices)
    return data, indices, indptr

def hstack_csc_cols(mats):
    if not isinstance(mats, (list,tuple)):
        mats = list(mats)
    assert len(mats)
    assert all(isinstance(mat, sp.csc_matrix) for mat in mats)
    assert all(mat.shape[1] == 1 for mat in mats)
    n_rows = mats[0].shape[0]
    assert all(mat.shape[0] == n_rows for mat in mats)
    n_cols = len(mats)
    state = statck_sparse_1d(mats)
    return unsafe_csc_matrix(state, (n_rows, n_cols))

def vstack_csr_rows(mats):
    if not isinstance(mats, (list,tuple)):
        mats = list(mats)
    assert len(mats)
    assert all(isinstance(mat, sp.csr_matrix) for mat in mats)
    assert all(mat.shape[0] == 1 for mat in mats)
    n_cols = mats[0].shape[1]
    if not all(mat.shape[1] == n_cols for mat in mats):
        raise ValueError("multiple numbers of columns in each row %s" %
                         ' '.join('%d' % i for i in
                                  set(mat.shape[1] == n_cols for mat in mats)))

    n_rows = len(mats)
    state = statck_sparse_1d(mats)
    return unsafe_csr_matrix(state, (n_rows, n_cols))

def _vstack_csc_cols(mats):
    acc_data = []
    acc_indices = []
    n_rows = 0

    for mat in mats:
        acc_data.append(mat.data)
        acc_indices.append(mat.indices + n_rows)
        n_rows += mat.shape[0]

    data = np.concatenate(acc_data)
    indices = np.concatenate(acc_indices).astype(np.intc)
    indptr = np.array([0, len(data)], dtype=np.intc)

    return unsafe_csc_matrix((data, indices, indptr), (n_rows, 1))

def vstack_csc_cols(mats):
    if not isinstance(mats, (list,tuple)):
        mats = list(mats)
    assert len(mats)
    assert all(isinstance(mat, sp.csc_matrix) for mat in mats)
    assert all(mat.shape[1] == 1 for mat in mats)
    return _vstack_csc_cols(mats)

def stack_sprase_mats(mats):
    acc_data = []
    acc_indptr = [[0]]
    acc_indices = []
    sum_indices = 0
    for mat in mats:
        acc_data.append(mat.data)
        acc_indptr.append(mat.indptr[1::] + sum_indices)
        acc_indices.append(mat.indices)
        sum_indices += len(mat.indices)
    data = np.concatenate(acc_data)
    indptr = np.concatenate(acc_indptr).astype(np.intc)
    indices = np.concatenate(acc_indices).astype(np.intc)
    return data, indices, indptr

def hstack_csc_mats(mats):
    if not isinstance(mats, (list,tuple)):
        mats = list(mats)
    assert len(mats)
    assert all(isinstance(mat, sp.csc_matrix) for mat in mats)
    n_rows = mats[0].shape[0]
    assert all(mat.shape[0] == n_rows for mat in mats)
    n_cols = sum(mat.shape[1] for mat in mats)
    state = stack_sprase_mats(mats)
    return unsafe_csc_matrix(state, (n_rows, n_cols))

def vstack_csr_mats(mats):
    if not isinstance(mats, (list,tuple)):
        mats = list(mats)
    assert len(mats)
    assert all(isinstance(mat, sp.csr_matrix) for mat in mats)
    n_cols = mats[0].shape[1]
    assert all(mat.shape[1] == n_cols for mat in mats)
    n_rows = sum(mat.shape[0] for mat in mats)
    state = stack_sprase_mats(mats)
    return unsafe_csr_matrix(state, (n_rows, n_cols))

def vstack_csc_mats(mats):
    if not isinstance(mats, (list,tuple)):
        mats = list(mats)
    assert len(mats)
    assert all(isinstance(mat, sp.csc_matrix) for mat in mats)
    n_cols = mats[0].shape[1]
    assert all(mat.shape[1] == n_cols for mat in mats)

    cols = zip(*(get_csc_cols(mat) for mat in mats))
    assert len(cols) == n_cols
    return hstack_csc_cols([_vstack_csc_cols(col)
                            for col in cols])


def get_select_compressed_elements(m, inxs, cls, tp):
    if tp == 'col':
        n = m.shape[0]
        shape = (n, 1)
    elif tp == 'row':
        n = m.shape[1]
        shape = (1, n)
    else:
        raise ValueError

    acc = []
    for i in inxs:
        start, stop = m.indptr[i:i+2]
        data = m.data[start:stop:]
        indices = m.indices[start:stop]
        indptr = np.array([0, len(data)], np.intc)
        acc.append(cls((data, indices, indptr),
                       shape))
    assert len(acc)
    return acc

def get_select_csc_cols(m, inxs):
    assert isinstance(m, sp.csc_matrix)
    return get_select_compressed_elements(m, inxs, unsafe_csc_matrix, 'col')

def get_select_csr_rows(m, inxs):
    assert isinstance(m, sp.csr_matrix)
    return get_select_compressed_elements(m, inxs, unsafe_csr_matrix, 'row')

def get_csc_cols(m):
    return get_select_csc_cols(m, xrange(m.shape[1]))

def get_csr_rows(m):
    return get_select_csr_rows(m, xrange(m.shape[0]))


def remove_csr_rows(mat, inxs):
    if isinstance(inxs, (int,long)):
        inxs = [inxs]
    inxs = np.asarray(inxs)
    assert len(inxs.shape) == 1
    if inxs.shape[0] == 0:
        return mat.copy()

    assert len(inxs.shape) == 1
    assert (np.diff(inxs) > 0).all()
    assert (0 <= inxs).all()
    assert (inxs <= mat.shape[0]).all()

    data = mat.data
    indices = mat.indices
    indptr = mat.indptr

    acc_data = []
    acc_indices = []
    acc_indptr = []
    last_row = None
    last_inx = 0
    indptr_offset = 0

    for i,inx in enumerate(inxs):

        row_start = indptr[inx]
        row_end = indptr[inx+1]

        acc_data.append(data[last_row:row_start])
        acc_indices.append(indices[last_row:row_start:])
        acc_indptr.append(indptr[last_inx:inx:] - indptr_offset)
        indptr_offset += row_end - row_start

        last_row = row_end
        last_inx = inx+1

    acc_data.append(data[last_row::])
    acc_indices.append(indices[last_row::])
    acc_indptr.append(indptr[last_inx::] - indptr_offset)

    data = np.concatenate(acc_data)
    indices = np.concatenate(acc_indices)
    indptr = np.concatenate(acc_indptr)

    return sp.csr_matrix((data, indices, indptr),
                         (mat.shape[0] - len(inxs), mat.shape[1]))
