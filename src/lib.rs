mod gf2p8;
mod poly_11d;

use gf2p8::bit_matrix::BitMatrix;
use gf2p8::generic::{CantorBasisLut, Gf2p8Lut};

// Forward FFT (Decimation in Frequency)
pub fn fft_recursive(shards: &mut [&mut [u8]], twiddles: &[BitMatrix]) {
    if shards.len() <= 1 {
        return;
    }
    let half = shards.len() / 2;
    let mat = twiddles[0];
    let (top, bottom) = shards.split_at_mut(half);

    // Conquer
    for i in 0..half {
        for j in 0..top[i].len() {
            // Forward: top = top + bottom, bottom = bottom + mat(top)
            top[i][j] ^= bottom[i][j];
            bottom[i][j] ^= mat.apply(top[i][j]);
        }
    }

    // Divide
    fft_recursive(top, &twiddles[1..]);
    fft_recursive(bottom, &twiddles[1..]);
}

/// Inverse FFT (Decimation in Time)
pub fn ifft_recursive(shards: &mut [&mut [u8]], twiddles: &[BitMatrix]) {
    if shards.len() <= 1 {
        return;
    }
    let half = shards.len() / 2;
    let (top, bottom) = shards.split_at_mut(half);

    // Divide
    ifft_recursive(top, &twiddles[1..]);
    ifft_recursive(bottom, &twiddles[1..]);

    // Conquer
    let mat = twiddles[0];
    for i in 0..half {
        for j in 0..top[i].len() {
            // Inverse: Undo bottom change, then undo top change
            bottom[i][j] ^= mat.apply(top[i][j]);
            top[i][j] ^= bottom[i][j];
        }
    }
}

/// M is the number of data shards and is a power of 2 <= 128.
pub fn encode<const M: usize, G: Gf2p8Lut>(
    data: &[&[u8]],           // M data shards
    parity: &mut [&mut [u8]], // M parity shards
    twiddles: &[BitMatrix],   // log2 M + 1 layers [Mat(vM), ..., Mat(v0)]
) {
    // Move data into the parity slots
    for i in 0..M {
        parity[i].copy_from_slice(data[i]);
    }

    // Interpolate
    // Treat the data as evaluations and find the polynomial coefficients.
    // Use layers 1.. (the M-shard subspace twiddles).
    ifft_recursive(parity, &twiddles[1..]);

    // The Bridge (Layer 0 / Stage 0)
    // This is the interaction between the data subspace and parity subspace. The widest butterfly.
    let twiddle = twiddles[0];
    for i in 0..M {
        for j in 0..parity[i].len() {
            // Only modify the parity side
            parity[i][j] = twiddle.apply(parity[i][j]);
        }
    }

    // Evaluate
    // Convert the modified coefficients into actual parity evaluations.
    fft_recursive(parity, &twiddles[1..]);
}

/// N is the sum of the number of data and parity shards and is a power of 2 <= 256.
pub fn reconstruct_in_place<const N: usize, G: Gf2p8Lut>(
    shards: &mut [&mut [u8]], // All N shards, some filled, some not
    is_erased: &[bool; N],    // Which indices are missing
    twiddles: &[BitMatrix],   // precomputed matrices
    basis: &impl CantorBasisLut<G>,
) {
    debug_assert!(
        (1..=8).any(|i| 2usize.pow(i) == N),
        "Number of shards should be a power of 2 up to 256"
    );

    let erased_indices: Vec<u8> = (0..N as u8).filter(|&i| is_erased[i as usize]).collect();

    // Pre-weighting
    for i in 0..N as u8 {
        if !is_erased[i as usize] {
            let weight = basis.eval_erasure_locator_poly_lut(i, &erased_indices);
            let mat = weight.into_bit_matrix();
            for byte in shards[i as usize].iter_mut() {
                *byte = mat.apply(*byte);
            }
        }
    }

    // Forward FFT + Inverse FFT
    // This "mixes" the known weighted values into the erasure slots.
    fft_recursive(shards, twiddles);
    ifft_recursive(shards, twiddles);

    // Post-weighting (correction)
    for i in 0..N as u8 {
        if is_erased[i as usize] {
            // In additive RS, the correction weight is 1 / E'(alpha_i).
            // For simple erasures, we evaluate the derivative polynomial.
            let corr_weight = basis
                .eval_erasure_locator_poly_lut(i, &erased_indices)
                .inv_lut();
            let mat = corr_weight.into_bit_matrix();
            for byte in shards[i as usize].iter_mut() {
                *byte = mat.apply(*byte);
            }
        }
    }
}

pub fn reconstruct_systematic<const N: usize, G: Gf2p8Lut>(
    received: &[(u8, &[u8])],
    workspace: &mut [&mut [u8]],
    twiddles: &[BitMatrix], // e.g., &basis.twiddle_factors[2..]
    basis: &impl CantorBasisLut<G>,
) -> bool {
    let m = N / 2;
    let mut is_erased = [true; N];
    for &(idx, _) in received {
        if (idx as usize) < N {
            is_erased[idx as usize] = false;
        }
    }

    // Identify exactly M erasures
    let mut erasures = Vec::with_capacity(m);
    for i in 0..N {
        if is_erased[i] {
            erasures.push(i as u8);
        }
    }
    if erasures.len() > m {
        return false;
    }
    let mut pad = 0;
    while erasures.len() < m {
        if !is_erased[pad] && !erasures.contains(&(pad as u8)) {
            erasures.push(pad as u8);
        }
        pad += 1;
    }

    // Weight knowns by L(alpha_i), erasures/dummies to 0
    for i in 0..N {
        if !erasures.contains(&(i as u8)) {
            let w = basis.eval_erasure_locator_poly_lut(i as u8, &erasures);
            w.scale_shard(workspace[i]);
        } else {
            workspace[i].fill(0);
        }
    }

    // Transform to find the syndrome
    let (data_part, parity_part) = workspace.split_at_mut(m);
    ifft_recursive(data_part, &twiddles[1..]);
    ifft_recursive(parity_part, &twiddles[1..]);

    let bridge = twiddles[0];
    let bridge_inv = bridge.inv().expect("Bridge not invertible");

    for i in 0..m {
        for j in 0..data_part[i].len() {
            // S = Parity ^ Bridge(Data)
            let syn = parity_part[i][j] ^ bridge.apply(data_part[i][j]);

            // Overwrite coefficients with Syndrome (Error)
            // Error_Data = Bridge_Inv(S), Error_Parity = S
            data_part[i][j] = bridge_inv.apply(syn);
            parity_part[i][j] = syn;
        }
    }

    fft_recursive(data_part, &twiddles[1..]);
    fft_recursive(parity_part, &twiddles[1..]);

    // Final un-weight and Restore
    for i in 0..N {
        let weight = basis.eval_erasure_locator_poly_lut(i as u8, &erasures);
        let inv_weight = weight.inv_lut();

        if is_erased[i] {
            inv_weight.scale_shard(workspace[i]);
        } else {
            // Restore from received to ensure the known data is original
            for &(idx, data) in received {
                if idx as usize == i {
                    workspace[i].copy_from_slice(data);
                    break;
                }
            }
        }
    }
    true
}

/*
/// N is the sum of the number of data and parity shards and is a power of 2 <= 256.
pub fn reconstruct_systematic<const N: usize, G: Gf2p8Lut>(
    received: &[(u8, &[u8])],    // received shards with their indices
    workspace: &mut [&mut [u8]], // pre-allocated N shards
    twiddles: &[BitMatrix],
    basis: &impl CantorBasisLut<G>,
) -> bool {
    debug_assert!(
        (1..=8).any(|i| 2usize.pow(i) == N),
        "Number of shards should be a power of 2 up to 256"
    );

    let shard_len = workspace[0].len();

    // Identify which indices are missing (needed for locator polynomial)
    let mut is_erased = [true; N];
    for &(i, _) in received {
        is_erased[i as usize] = false;
    }

    let erased_indices: Vec<u8> = (0..N as u8).filter(|&i| is_erased[i as usize]).collect();

    // Prepare the workspace: map received shards & pre-weight
    // We only touch the indices we actually received.
    for &(i, data) in received {
        // Copy original to workspace
        workspace[i as usize].copy_from_slice(data);

        // Apply weight E(alpha_i)
        let weight = basis.eval_erasure_locator_poly_lut(i, &erased_indices);
        let mat = weight.into_bit_matrix();
        for byte in workspace[i as usize].iter_mut() {
            *byte = mat.apply(*byte);
        }
    }

    // Zero-fill the gaps in the workspace
    for &i in &erased_indices {
        workspace[i as usize].fill(0);
    }

    // Transform: forward FFT followed by Inverse FFT
    fft_recursive(workspace, twiddles);
    ifft_recursive(workspace, twiddles);

    // // Integrity check: witness test
    // // For a received shard, the workspace now contains (original * E(alpha_w))
    // let (witness_i, witness_original_data) = received[0];
    // let weight = basis
    //     .eval_erasure_locator_poly_lut(witness_i, &erased_indices)
    //     .inv_lut();
    // let mat = weight.into_bit_matrix();

    // for i in 0..shard_len {
    //     let recovered_byte = mat.apply(workspace[witness_i as usize][i]);
    //     if recovered_byte != witness_original_data[i] {
    //         return false;
    //     }
    // }

    // Post-weight: recover all erased shards
    for &i in &erased_indices {
        // Apply correction weight 1 / E'(alpha_idx)
        let corr_weight = basis
            .eval_erasure_locator_poly_lut(i, &erased_indices)
            .inv_lut();

        let mat = corr_weight.into_bit_matrix();
        for byte in workspace[i as usize].iter_mut() {
            *byte = mat.apply(*byte);
        }
    }

    true
}
*/
