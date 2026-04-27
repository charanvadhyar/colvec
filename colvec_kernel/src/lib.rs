use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use numpy::ndarray::Array2;

/// Apply PQ lookup tables to corpus codes using AVX2 SIMD.
///
/// lookups: [num_q, M, K]    f32  — similarity of query chunks to codebook entries
/// codes_t: [M, N_tokens]    u8   — transposed codes (chunk-major layout)
///
/// Returns sim: [num_q, N_tokens] f32
///
/// Inner loop processes 8 tokens simultaneously using AVX2 gather+add.
#[pyfunction]
fn apply_pq_lookups<'py>(
    py: Python<'py>,
    lookups: PyReadonlyArray3<'py, f32>,
    codes_t: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {

    let lookups  = lookups.as_array();
    let codes_t  = codes_t.as_array();

    let num_q    = lookups.shape()[0];
    let m_chunks = lookups.shape()[1];
    let n_tokens = codes_t.shape()[1];

    let mut sim = vec![0.0f32; num_q * n_tokens];

    // Get raw pointers for unsafe SIMD work
    // Safety: we only read within bounds we've verified above
    let sim_ptr     = sim.as_mut_ptr();
    let codes_ptr   = codes_t.as_ptr();
    let lookups_ptr = lookups.as_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: we just checked AVX2 is available at runtime
            unsafe {
                apply_avx2(
                    sim_ptr, lookups_ptr, codes_ptr,
                    num_q, m_chunks, n_tokens,
                );
            }

            let result = Array2::from_shape_vec((num_q, n_tokens), sim).unwrap();
            return Ok(result.into_pyarray_bound(py).into());
        }
    }

    // Fallback: scalar path (same as naive version, for non-AVX2 CPUs)
    for q in 0..num_q {
        for m in 0..m_chunks {
            let chunk_codes = codes_t.row(m);
            let lookup_row  = lookups.slice(numpy::ndarray::s![q, m, ..]);
            let sim_row     = &mut sim[q * n_tokens .. (q + 1) * n_tokens];
            for n in 0..n_tokens {
                let code = chunk_codes[n] as usize;
                sim_row[n] += lookup_row[code];
            }
        }
    }

    let result = Array2::from_shape_vec((num_q, n_tokens), sim).unwrap();
    Ok(result.into_pyarray_bound(py).into())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_avx2(
    sim_ptr:     *mut f32,
    lookups_ptr: *const f32,
    codes_ptr:   *const u8,
    num_q:    usize,
    m_chunks: usize,
    n_tokens: usize,
) {
    use std::arch::x86_64::*;

    // K=256 entries per lookup row, M chunks, num_q query tokens
    // lookups layout: [num_q, M, K] row-major
    // codes_t layout: [M, N_tokens] row-major
    // sim layout:     [num_q, N_tokens] row-major

    let k = 256usize;   // always 256 — one byte per code

    for q in 0..num_q {
        for m in 0..m_chunks {
            // Pointer to lookup_row = lookups[q, m, 0]
            let lookup_row: *const f32 = lookups_ptr
                .add(q * m_chunks * k + m * k);

            // Pointer to chunk_codes = codes_t[m, 0]
            let chunk_codes: *const u8 = codes_ptr
                .add(m * n_tokens);

            // Pointer to sim_row = sim[q, 0]
            let sim_row: *mut f32 = sim_ptr
                .add(q * n_tokens);

            // Process 8 tokens at a time with AVX2
            let n_simd = (n_tokens / 8) * 8;   // round down to multiple of 8

            let mut n = 0usize;
            while n < n_simd {
                // Load 8 uint8 codes and widen to int32 for gather
                // codes_ptr[n..n+8] are the codebook indices for 8 tokens
                let c0 = *chunk_codes.add(n)     as i32;
                let c1 = *chunk_codes.add(n + 1) as i32;
                let c2 = *chunk_codes.add(n + 2) as i32;
                let c3 = *chunk_codes.add(n + 3) as i32;
                let c4 = *chunk_codes.add(n + 4) as i32;
                let c5 = *chunk_codes.add(n + 5) as i32;
                let c6 = *chunk_codes.add(n + 6) as i32;
                let c7 = *chunk_codes.add(n + 7) as i32;

                // Pack the 8 indices into an AVX2 integer vector
                let indices = _mm256_set_epi32(c7, c6, c5, c4, c3, c2, c1, c0);

                // Gather: load 8 floats from lookup_row at the 8 indices
                // _mm256_i32gather_ps(base, indices, scale)
                //   base:    pointer to lookup_row[0]
                //   indices: 8 × i32 indices
                //   scale:   4 (sizeof f32) — indices are in elements, not bytes
                let gathered = _mm256_i32gather_ps(
                    lookup_row,
                    indices,
                    4,
                );

                // Load 8 current sim values, add gathered values, store back
                let sim_vec = _mm256_loadu_ps(sim_row.add(n));
                let result  = _mm256_add_ps(sim_vec, gathered);
                _mm256_storeu_ps(sim_row.add(n), result);

                n += 8;
            }

            // Scalar tail: handle remaining tokens (n_tokens % 8 leftovers)
            while n < n_tokens {
                let code = *chunk_codes.add(n) as usize;
                *sim_row.add(n) += *lookup_row.add(code);
                n += 1;
            }
        }
    }
}

#[pymodule]
fn colvec_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_pq_lookups, m)?)?;
    Ok(())
}