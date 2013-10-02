
import unittest

import numpy as np
import scipy.sparse as sp
import sputil as spu


def create_unique_random_number(mn, mx, N):
    assert N < mx-mn
    acc = set()
    while len(acc) < N:
        i = np.random.randint(mn, mx)
        if i not in acc:
            acc.add(i)
    return acc

def make_sparse_matrix(N_rows=1, N_cols=1, fill_frac=0.01,
                       dtype=float, format='csr'):
    dtype = np.dtype(dtype)
    m = sp.dok_matrix((N_rows, N_cols), dtype=dtype)
    N_elements = N_rows * N_cols
    n_fill = int(round(fill_frac * N_elements))
    fill_indices_1d = create_unique_random_number(0, N_elements, n_fill)

    for inx_1d in fill_indices_1d:
        row_i,col_i = divmod(inx_1d, N_cols)
        m[row_i, col_i] = np.random.randint(-1000000, 1000000)
    assert m.getnnz() == n_fill
    return m.asformat(format)

def make_sparse_row(N_cols, format='csr', **kwds):
    return make_sparse_matrix(N_rows=1, N_cols=N_cols, format=format, **kwds)

def make_sparse_col(N_rows, format='csc', **kwds):
    return make_sparse_matrix(N_rows=N_rows, N_cols=1, format=format, **kwds)

def assert_eq_sparse_matrices(m1, m2):
    assert m1.shape == m2.shape
    assert np.all(m1.toarray() == m2.toarray())

def make_hstack_mat_set(N=100, N_rows=25, min_n_cols=20, max_n_cols=50, format='csc', **kwds):
    return [make_sparse_matrix(N_rows, np.random.randint(min_n_cols, max_n_cols),
                               format=format, **kwds)
            for _ in xrange(N)]

def make_vstack_mat_set(N=100, N_cols=25, min_n_rows=20, max_n_rows=50, format='csr', **kwds):
    return [make_sparse_matrix(np.random.randint(min_n_rows, max_n_rows), N_cols,
                               format=format, **kwds)
            for _ in xrange(N)]

class TestHstackCSCCols(unittest.TestCase):

    def testhstack_csc_cols(self):
        cols = [make_sparse_col(1000) for _ in xrange(1000)]
        m1 = sp.hstack(cols)
        m2 = spu.hstack_csc_cols(cols)
        assert_eq_sparse_matrices(m1, m2)

class TestVstackCSRRows(unittest.TestCase):

    def testvstack_csr_rows(self):
        rows = [make_sparse_row(1000) for _ in xrange(1000)]
        m1 = sp.vstack(rows)
        m2 = spu.vstack_csr_rows(rows)
        assert_eq_sparse_matrices(m1, m2)

class TestHstackCSCMats(unittest.TestCase):

    def testhstack_csc_mats(self):
        mats = make_hstack_mat_set(format='csc')
        m1 = sp.hstack(mats)
        m2 = spu.hstack_csc_mats(mats)
        assert_eq_sparse_matrices(m1, m2)

class TestVstackCSRMats(unittest.TestCase):

    def testvstack_csr_mats(self):
        mats = make_vstack_mat_set(format='csr')
        m1 = sp.vstack(mats)
        m2 = spu.vstack_csr_mats(mats)
        assert_eq_sparse_matrices(m1, m2)

class TestVstackCSCCols(unittest.TestCase):

    def testvstack_csc_mats(self):
        mats = make_vstack_mat_set(format='csc', N_cols=1)
        m1 = sp.vstack(mats)
        m2 = spu.vstack_csc_cols(mats)
        assert_eq_sparse_matrices(m1, m2)

class TestVstackCSCMats(unittest.TestCase):

    def testvstack_csc_mats(self):
        mats = make_vstack_mat_set(format='csc')
        m1 = sp.vstack(mats)
        m2 = spu.vstack_csc_mats(mats)
        assert_eq_sparse_matrices(m1, m2)

class TestGetSelectCSCCols(unittest.TestCase):

    def testget_select_csc_cols(self):
        m = make_sparse_matrix(N_rows=100, N_cols=200, format='csc')
        indices = np.random.randint(0, m.shape[1], 50)
        cols = spu.get_select_csc_cols(m, indices)
        assert len(cols) == len(indices)
        for inx,col in zip(indices, cols):
            assert_eq_sparse_matrices(col, m.getcol(inx))


class TestGetSelectCSCRRows(unittest.TestCase):

    def testget_select_csr_rows(self):
        m = make_sparse_matrix(N_rows=200, N_cols=100, format='csr')
        indices = np.random.randint(0, m.shape[0], 50)
        rows = spu.get_select_csr_rows(m, indices)
        assert len(rows) == len(indices)
        for inx,row in zip(indices, rows):
            assert_eq_sparse_matrices(row, m.getrow(inx))



class TestGetCSCCols(unittest.TestCase):

    def testget_csc_cols(self):
        m = make_sparse_matrix(N_rows=100, N_cols=200, format='csc')
        indices = range(m.shape[1])
        cols = spu.get_csc_cols(m)
        assert len(cols) == len(indices)
        for inx,col in zip(indices, cols):
            assert_eq_sparse_matrices(col, m.getcol(inx))


class TestGetCSCRRows(unittest.TestCase):

    def testget_csr_rows(self):
        m = make_sparse_matrix(N_rows=200, N_cols=100, format='csr')
        indices = range(m.shape[0])
        rows = spu.get_csr_rows(m)
        assert len(rows) == len(indices)
        for inx,row in zip(indices, rows):
            assert_eq_sparse_matrices(row, m.getrow(inx))


class TestRemoveCSRRows(unittest.TestCase):

    def testremove_csr_rows(self):
        N_rows = 1000
        remove_frac = 0.1
        rows = [make_sparse_row(1000) for _ in xrange(N_rows)]

        remove_indices = np.arange(N_rows)
        np.random.shuffle(remove_indices)
        remove_indices = remove_indices[:int(remove_frac * N_rows):]
        remove_indices = np.sort(remove_indices)
        remove_indices_set = set(remove_indices)

        mat = spu.vstack_csr_rows(rows)
        mat_wo_rows = spu.vstack_csr_rows([r for i,r in enumerate(rows)
                                          if i not in remove_indices_set])

        mat_remove_rows = spu.remove_csr_rows(mat, remove_indices)

        assert_eq_sparse_matrices(mat_wo_rows, mat_remove_rows)


__name__ == '__main__' and unittest.main()
