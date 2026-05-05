import numpy as np
import pickle

with open('data/scifact_pq_m32.pkl', 'rb') as f:
    pq_data = pickle.load(f)
codes = pq_data['codes']
codes_t = np.ascontiguousarray(codes.T)
print('codes shape:', codes.shape)
print('codes_t shape:', codes_t.shape)
print('codes_t C-contiguous:', codes_t.flags['C_CONTIGUOUS'])
print('codes dtype:', codes.dtype)
print('codes_t dtype:', codes_t.dtype)