"""
This module implements the soft decoder of binary LDPC codes.
C++ implementation is stored in impl directory.
cTypes are used to execute decoder from python
"""

# This file is part of the simulator_awgn_python distribution
# https://github.com/and-kirill/ldpc_soft_py/.
# Copyright (c) 2023 Kirill Andreev.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import ctypes
import os
import numpy as np


class Alist:
    """
    Alist parser for binary matrices.
    See http://www.inference.org.uk/mackay/codes/alist.html for more details
    """
    @staticmethod
    def write(matr, filename):
        """
        Write matrix to file
        :param matr: 2D np.array to be processed
        :param filename: Filename to save (utf-8 encoded)
        """
        column_weights = np.sum(matr > 0, axis=0)
        row_weights = np.sum(matr > 0, axis=1)
        row_max = np.max(row_weights)
        col_max = np.max(column_weights)

        with open(filename, 'w', encoding='utf-8') as file:
            # Print shape
            print(f'{matr.shape[1]} {matr.shape[0]}', file=file)
            # Print maximum row weight and column weight
            print(f'{col_max} {row_max}', file=file)
            # Print column weights
            print(Alist.to_string(column_weights), file=file)
            # Print row weights
            print(Alist.to_string(row_weights), file=file)

            for i in range(matr.shape[1]):
                print(Alist.to_string(np.nonzero(matr[:, i])[0] + 1, col_max), file=file)
            for i in range(matr.shape[0]):
                print(Alist.to_string(np.nonzero(matr[i, :])[0] + 1, row_max), file=file)

    @staticmethod
    def read(filename):
        """
        Read aliist file and fill 2D numpy array
        :param filename: file with alist data (utf-8 encoded)
        :return: 2D binary np.array (np.uint8 type)
        """
        with open(filename, 'r', encoding='utf-8') as file:
            matrix_size = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
            matr = np.zeros((matrix_size[1], matrix_size[0]), dtype=np.uint)
            max_counts = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
            row_weights = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
            col_weights = np.fromstring(file.readline(), sep=' ', dtype=np.uint)

            for i in range(matr.shape[1]):
                idx = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
                assert len(idx) == max_counts[0]
                # Remove zeros
                idx = idx[idx > 0] - 1
                assert len(idx) == row_weights[i]
                # Assign matrix elements
                matr[idx, i] = 1

            # The remaining of the file is redundant. Just sanity checks below
            for i in range(matr.shape[0]):
                idx = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
                assert len(idx) == max_counts[1]
                idx = idx[idx > 0] - 1
                assert len(idx) == col_weights[i]
                assert np.sum(matr[i, :]) == len(idx)
                assert np.sum(matr[i, idx]) == len(idx)

        return matr

    @staticmethod
    def to_string(np_arr, length=None):
        """
        Convert numpy array to space-separated string
        :param np_arr:
        :param length: total length of the array (zero-padding if required)
        :return: string (space separated, as required by alist formart)
        """
        if length:
            np_arr = np.hstack([np_arr, np.array([0] * (length - len(np_arr)))])
        return ' '.join(map(str, np_arr.astype(np.uint).tolist()))


def invert_permutation(perm_forward):
    """
    Construct the inverse permutation.
    Note that matlab-like style of the inverse permutation p[idx]=p may not work for numpy
    """
    perm_reverse = np.empty(perm_forward.size, perm_forward.dtype)
    perm_reverse[perm_forward] = np.arange(perm_forward.size)
    return perm_reverse


def generator_from_pcm(pcm_rdonly):
    """
    Construct the generator matrix from the parity check matrix
    return: generator matrix (np.array, uint8) and information bits indices
    """
    pcm = pcm_rdonly.copy()  # Make a copy, row combinations will be performed
    n_rows, n_cols = pcm.shape
    col_ind = 0
    row_ind = 0
    eye_idx = []
    while row_ind < n_rows:
        # Find the first non-zero entry in column below <index> value
        nz_idx = np.argwhere(pcm[row_ind:, col_ind] == 1).reshape(-1)
        if len(nz_idx) == 0:
            col_ind += 1
            continue
        # Swap rows
        swap_ind = nz_idx[0] + row_ind
        pcm[[row_ind, swap_ind], :] = pcm[[swap_ind, row_ind], :]

        for i in np.argwhere(pcm[:, col_ind] == 1).reshape(-1):
            if row_ind == i:
                continue
            pcm[i, :] = np.mod(pcm[i, :] + pcm[row_ind, :], 2)
        eye_idx.append(col_ind)
        row_ind += 1
        col_ind = row_ind

    pc_idx = np.setdiff1d(np.arange(n_cols), eye_idx)  # Parity check indices
    all_idx = np.hstack([eye_idx, pc_idx])  # All indices (permuted)
    gen_mtx = np.hstack([pcm.copy()[:, pc_idx].T, np.eye(n_cols - n_rows).astype(np.uint8)])
    return gen_mtx[:, invert_permutation(all_idx)], all_idx[n_rows:]


# C++ implementation: compilation, linking, and execution routines
if os.name == 'nt':
    LIB_PATH = 'ldpc_decoder.dll'
else:
    LIB_PATH = 'ldpc_decoder.so'


def lib_compile():
    """
    Compile the C++ SCL decoder implementation
    """
    wdir = os.path.dirname(__file__)
    # if os.path.isfile(LIB_PATH):
    #     return
    src = ['ldpc', 'ldpc_ctypes']
    src_abs = [os.path.join(wdir, s) for s in src]
    if os.name == 'nt':
        if not os.popen('where g++').read():
            raise RuntimeError('g++ not found.')
    else:
        if not os.popen('which g++').read():
            raise RuntimeError('g++ not found.')

    for src_file in src_abs:
        os.system(f'g++ -O3 -fPIC -c -o {src_file}.o {src_file}.cpp')
    os.system(
        'g++ -shared -o ' +
        os.path.join(wdir, LIB_PATH) + ' ' +
        ''.join([s + '.o ' for s in src_abs])
    )
    obj_files = os.path.join(wdir, '*.o')
    if os.name == 'nt':
        os.system(f'del {obj_files}')
    else:
        os.system(f'rm {obj_files}')


def load_lib():
    """
    Load shared library
    """
    wdir = os.path.abspath(os.path.dirname(__file__))
    lib = ctypes.CDLL(os.path.join(wdir, LIB_PATH))
    # Init decoder function
    lib.init_ldpc.restype = ctypes.c_void_p
    lib.init_ldpc.argtypes = [
        ctypes.c_char_p
    ]
    # Decode functions
    lib.decode_soft.restype = None
    lib.decode_soft.argtypes = [
        ctypes.c_uint,                            # Command
        ctypes.c_void_p,                          # LDPC pointer
        np.ctypeslib.ndpointer(dtype=np.double),  # Channel LLR
        ctypes.c_uint,                            # Decoding iterations
        np.ctypeslib.ndpointer(dtype=np.double),  # Output LLRs
        np.ctypeslib.ndpointer(dtype=np.uint32),  # Row sequence
        np.ctypeslib.ndpointer(dtype=np.double),  # Scale array
        np.ctypeslib.ndpointer(dtype=np.double),  # Offset array
        np.ctypeslib.ndpointer(dtype=np.double),  # Output intermediate LLRs
    ]
    # Destroy LDPC object
    lib.free_ldpc.restype = None
    lib.free_ldpc.argtypes = [
        ctypes.c_void_p
    ]
    return lib


class LdpcDecoder:
    """
    This class creates the LDPC decoder instance given alist file
    and provides all LDPC decoding routines
    """

    def __init__(self, alist_filename, n_iterations, llr_scale=1.0):
        self.shared_object = load_lib()
        self.n_checks, self.block_len = Alist.read(alist_filename).shape
        self.ldpc_ptr = self.shared_object.init_ldpc(alist_filename.encode())

        if not self.ldpc_ptr:
            raise RuntimeError('Failed to initialize decoder.')

        self.n_iterations = n_iterations
        self.scales = np.ones(self.block_len,) * llr_scale
        self.offsets = np.zeros(self.block_len,)
        self.row_sequence = np.arange(self.n_checks - 1, -1, -1).astype(np.uint32)

    def __del__(self):
        self.shared_object.free_ldpc(self.ldpc_ptr)

    def sum_product(self, llr_in):
        """
        Run sum-product decoder (layered implementation)
        """
        # 0 = sum-product
        return self.__decode_soft(0, llr_in)

    def layered_min_sum(self, llr_in):
        """
        Run layered min-sum decoder
        """
        # 3 = layered min-sum
        return self.__decode_soft(3, llr_in)

    def min_sum(self, llr_in):
        """
        Run min-sum decoder
        """
        # 2 = min-sum
        return self.__decode_soft(2, llr_in)

    def __decode_soft(self, decoder_type, llr_in):
        """
        Run C++ implementation
        """
        assert self.block_len == len(llr_in)
        llr_out = np.zeros(self.block_len, dtype=np.float64)
        llr_out_intermediate = np.zeros(len(llr_in)*self.n_iterations, dtype=np.float64)
        self.shared_object.decode_soft(
            decoder_type,
            self.ldpc_ptr,
            llr_in,
            self.n_iterations,
            llr_out,
            self.row_sequence,
            self.scales,
            self.offsets,
            llr_out_intermediate
        )
        return llr_out, llr_out_intermediate.reshape((self.n_iterations, len(llr_in)))


if __name__ == '__main__':
    lib_compile()