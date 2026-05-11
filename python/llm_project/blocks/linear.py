"""FD (folded-diagonal) linear layer: encode, encrypt, and matvec.

Python port of the orchestration logic in src/linear.cu.
"""

import sys
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")

import numpy as np
import pyPhantom as phantom


class EncodedMatrixFD:
    """FD-packed matrix as a list of SingleChainPlaintexts."""
    __slots__ = ("chunks", "num_rows", "num_cols", "cols_per_chunk")

    def __init__(self, chunks, num_rows, num_cols, cols_per_chunk):
        self.chunks = chunks              # list[phantom.single_chain_plaintext]
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cols_per_chunk = cols_per_chunk

    @property
    def num_chunks(self):
        return len(self.chunks)


class EncryptedVectorFD:
    """FD-packed vector as a list of ciphertexts."""
    __slots__ = ("chunks", "num_chunks", "vector_dim", "feature_dim")

    def __init__(self, chunks, vector_dim, feature_dim):
        self.chunks = chunks              # list[phantom.ciphertext]
        self.num_chunks = len(chunks)
        self.vector_dim = vector_dim
        self.feature_dim = feature_dim


def encode_matrix_fd(ctx, encoder, matrix, num_rows, num_cols, scale):
    """Encode a (num_rows x num_cols) matrix into FD-packed SingleChainPlaintexts."""
    if num_rows == 0 or num_cols == 0:
        raise ValueError("encode_matrix_fd: dimensions must be non-zero")
    matrix = list(matrix)
    if len(matrix) != num_rows * num_cols:
        raise ValueError("encode_matrix_fd: matrix size mismatch")

    num_slots = encoder.slot_count()
    if num_rows > num_slots:
        raise ValueError("encode_matrix_fd: num_rows exceeds num_slots")
    if num_slots % num_rows != 0:
        raise ValueError("encode_matrix_fd: num_slots must be divisible by num_rows")

    cols_per_chunk = num_slots // num_rows
    num_chunks = (num_cols + cols_per_chunk - 1) // cols_per_chunk

    mat_np = np.array(matrix, dtype=np.float64).reshape(num_rows, num_cols)

    chunks = []
    for c in range(num_chunks):
        col_start = c * cols_per_chunk
        col_end = min(col_start + cols_per_chunk, num_cols)
        slots = np.zeros(num_slots, dtype=complex)
        for local_j in range(col_end - col_start):
            global_j = col_start + local_j
            for i in range(num_rows):
                slots[local_j * num_rows + i] = complex(mat_np[i, global_j], 0.0)
        scp = phantom.encode_single_chain_plaintext(ctx, encoder, slots, scale)
        chunks.append(scp)

    return EncodedMatrixFD(chunks, num_rows, num_cols, cols_per_chunk)


def encrypt_vector_fd(ctx, encoder, sk, vector, feature_dim, scale, chain_index):
    """Encrypt a real vector into FD-packed ciphertexts."""
    if feature_dim == 0 or len(vector) == 0:
        raise ValueError("encrypt_vector_fd: dimensions must be non-zero")
    vector = list(vector)
    num_slots = encoder.slot_count()
    if feature_dim > num_slots:
        raise ValueError("encrypt_vector_fd: feature_dim exceeds num_slots")
    if num_slots % feature_dim != 0:
        raise ValueError("encrypt_vector_fd: num_slots must be divisible by feature_dim")

    cols_per_chunk = num_slots // feature_dim
    num_chunks = (len(vector) + cols_per_chunk - 1) // cols_per_chunk

    vec_np = np.array(vector, dtype=np.float64)

    chunks = []
    for c in range(num_chunks):
        col_start = c * cols_per_chunk
        col_end = min(col_start + cols_per_chunk, len(vec_np))
        slots = np.zeros(num_slots, dtype=np.float64)
        for local_j in range(col_end - col_start):
            v = vec_np[col_start + local_j]
            slots[local_j * feature_dim : local_j * feature_dim + feature_dim] = v
        pt = encoder.encode_double_vector(ctx, slots, scale, chain_index)
        ct = sk.encrypt_symmetric(ctx, pt)
        chunks.append(ct)

    return EncryptedVectorFD(chunks, len(vector), feature_dim)


def matvec_fd_required_steps(feature_dim, cols_per_chunk):
    """Rotation steps for the inner-sum phase of multiply_matrix_vector_fd."""
    steps = []
    if feature_dim == 0 or cols_per_chunk <= 1:
        return steps
    max_slots = cols_per_chunk * feature_dim
    stride = feature_dim
    while stride < max_slots:
        steps.append(int(stride))
        stride <<= 1
    return steps


def multiply_matrix_vector_fd(ctx, gk, mat_fd, vec_fd):
    """FD matrix-vector multiply (multiply_plain + rotate + add, no ct x ct)."""
    mat_chunks = mat_fd.chunks
    vec_chunks = vec_fd.chunks
    num_chunks = len(mat_chunks)

    if num_chunks == 0:
        raise ValueError("multiply_matrix_vector_fd: empty matrix")
    if num_chunks != len(vec_chunks):
        raise ValueError("multiply_matrix_vector_fd: chunk count mismatch")
    if mat_fd.num_rows != vec_fd.feature_dim:
        raise ValueError("multiply_matrix_vector_fd: feature_dim mismatch")
    if mat_fd.num_cols != vec_fd.vector_dim:
        raise ValueError("multiply_matrix_vector_fd: vector_dim mismatch")

    num_rows = mat_fd.num_rows
    cols_per_chunk = mat_fd.cols_per_chunk

    target_ci = vec_chunks[0].chain_index()
    pt0 = phantom.expand_single_chain_to_full(ctx, mat_chunks[0], target_ci)
    acc = phantom.multiply_plain(ctx, vec_chunks[0], pt0)

    for c in range(1, num_chunks):
        ptc = phantom.expand_single_chain_to_full(ctx, mat_chunks[c], target_ci)
        prod = phantom.multiply_plain(ctx, vec_chunks[c], ptc)
        acc = phantom.add(ctx, acc, prod)

    max_slots = cols_per_chunk * num_rows
    stride = num_rows
    while stride < max_slots:
        rotated = phantom.rotate(ctx, acc, int(stride), gk)
        acc = phantom.add(ctx, acc, rotated)
        stride <<= 1

    return phantom.rescale_to_next(ctx, acc)


def inner_sum_required_steps(block_size):
    """Galois steps for inner_sum: {1, 2, 4, ..., block_size/2}."""
    steps = []
    if block_size < 2:
        return steps
    stride = 1
    while stride < block_size:
        steps.append(int(stride))
        stride <<= 1
    return steps


def replicate_required_steps(period, num_slots):
    """Galois steps for replicate: {-period, -2*period, ..., -(num_slots/2)}."""
    if period == 0 or (period & (period - 1)) != 0:
        raise ValueError("replicate_required_steps: period must be a power of 2")
    if num_slots == 0 or (num_slots & (num_slots - 1)) != 0:
        raise ValueError("replicate_required_steps: num_slots must be a power of 2")
    if period > num_slots:
        raise ValueError("replicate_required_steps: period exceeds num_slots")
    steps = []
    stride = period
    while stride < num_slots:
        steps.append(-int(stride))
        stride <<= 1
    return steps
