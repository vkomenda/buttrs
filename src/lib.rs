mod gf2p8;
mod poly_11d;

use gf2p8::bit_matrix::BitMatrix;
use gf2p8::generic::{CantorBasisLut, Gf2p8Lut};

/// The Additive FFT Recursive Step
/// This divides the N shards into two N/2 square sub-problems.
pub fn fft_recursive(shards: &mut [&mut [u8]], twiddles: &[BitMatrix]) {
    let n = shards.len();
    if n <= 1 {
        return;
    }

    let half = n / 2;
    let (top, bottom) = shards.split_at_mut(half);

    // Recursive Step (Divide)
    fft_recursive(top, &twiddles[1..]);
    fft_recursive(bottom, &twiddles[1..]);

    // Butterfly Combine (Conquer)
    // In an additive FFT, this stage interacts rows of the top block with the bottom block.
    let mat = twiddles[0];

    for i in 0..half {
        for j in 0..top[i].len() {
            let u = top[i][j];
            let v = bottom[i][j];

            // Standard Additive Butterfly:
            // u' = u + v
            // v' = v * Twiddle + u
            let u_new = u ^ v;
            let v_weighted = mat.apply(v);
            let v_new = v_weighted ^ u; // Note: using 'u' here preserves entropy

            top[i][j] = u_new;
            bottom[i][j] = v_new;
        }
    }
}

pub fn ifft_recursive(shards: &mut [&mut [u8]], twiddles: &[BitMatrix]) {
    let n = shards.len();
    if n <= 1 {
        return;
    }

    let half = n / 2;
    let (top, bottom) = shards.split_at_mut(half);

    ifft_recursive(top, &twiddles[1..]);
    ifft_recursive(bottom, &twiddles[1..]);

    let mat = twiddles[0];
    for i in 0..half {
        for j in 0..top[i].len() {
            let u = top[i][j];
            let v = bottom[i][j];

            // Inverse butterfly logic
            let u_new = u ^ v;
            let v_weighted = mat.apply(u_new);
            top[i][j] = u_new;
            bottom[i][j] = v ^ v_weighted;
        }
    }
}

/// M is the number of data shards and is a power of 2 <= 128.
pub fn encode<const M: usize, G: Gf2p8Lut>(
    data: &[&[u8]],           // N data shards
    parity: &mut [&mut [u8]], // N parity shards
    twiddles: &[BitMatrix],   // log2 N layers [Mat(vn), ..., Mat(v0)]
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

    // Integrity check: witness test
    // For a received shard, the workspace now contains (original * E(alpha_w))
    let (witness_i, witness_original_data) = received[0];
    let weight = basis
        .eval_erasure_locator_poly_lut(witness_i, &erased_indices)
        .inv_lut();
    let mat = weight.into_bit_matrix();

    for i in 0..shard_len {
        let recovered_byte = mat.apply(workspace[witness_i as usize][i]);
        if recovered_byte != witness_original_data[i] {
            return false;
        }
    }

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
