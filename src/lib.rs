mod bit_matrix;
mod gf2p8;

use bit_matrix::BitMatrix;
use gf2p8::Gf2p8;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Gf2p8_11d(pub u8);

impl From<u8> for Gf2p8_11d {
    fn from(a: u8) -> Self {
        Self(a)
    }
}

impl From<Gf2p8_11d> for u8 {
    fn from(a: Gf2p8_11d) -> u8 {
        a.0
    }
}

impl Gf2p8 for Gf2p8_11d {
    const POLY: u16 = 0x11d;
}

/// The Additive FFT Recursive Step
/// This divides the N shards into two N/2 square sub-problems.
pub fn fft_recursive(shards: &mut [&mut [u8]], twiddles: &[BitMatrix]) {
    let n = shards.len();
    if n <= 1 {
        return;
    }

    let half = n / 2;

    // 1. Split into Top and Bottom square submatrices
    let (top, bottom) = shards.split_at_mut(half);

    // 2. Recursive Step (Divide)
    fft_recursive(top, &twiddles[1..]);
    fft_recursive(bottom, &twiddles[1..]);

    // 3. Butterfly Combine (Conquer)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_fft_4_shards() {
        // Data: 4 shards of 2 bytes each
        let mut s0 = [10, 20];
        let mut s1 = [30, 40];
        let mut s2 = [50, 60];
        let mut s3 = [70, 80];

        let mut shards: Vec<&mut [u8]> = vec![&mut s0, &mut s1, &mut s2, &mut s3];

        // For a 4-shard FFT, we need 2 levels of twiddle matrices.
        // We use constants 0x01 and 0x02 as a simple basis.
        let twiddles = vec![
            Gf2p8_11d::from(0x01).into_bit_matrix(),
            Gf2p8_11d::from(0x02).into_bit_matrix(),
        ];

        fft_recursive(&mut shards, &twiddles);

        println!("s0: {:?}, s1: {:?}, s2: {:?}, s3: {:?}", s0, s1, s2, s3);

        // Verify that all shards were modified
        assert_ne!(s0, [10, 20], "Shard 0 was not modified");
        assert_ne!(s1, [30, 40], "Shard 1 was not modified");
        assert_ne!(s2, [50, 60], "Shard 2 was not modified");
        assert_ne!(s3, [70, 80], "Shard 3 was not modified");
    }

    #[test]
    fn trace_of_1() {
        assert!(!Gf2p8_11d::from(1).trace());
    }

    #[test]
    fn test_cantor_basis_properties() {
        let basis = Gf2p8_11d::generate_cantor_basis();

        // 1. Check length: For GF(2^8), we expect a basis of 8 elements.
        assert_eq!(basis.len(), 8, "Basis should have 8 elements for GF(2^8)");

        // 2. Check the Chain Property: v_i^2 + v_i = v_{i-1}
        for i in 1..basis.len() {
            let v_curr = Gf2p8_11d::from(basis[i]);
            let v_prev = Gf2p8_11d::from(basis[i - 1]);

            // v^2 + v
            let lhs = v_curr.mul(v_curr).add(v_curr);
            assert_eq!(lhs, v_prev, "Chain property failed at index {}", i);
        }

        // 3. Check Trace Conditions
        // For the sequence to be extendable, Tr(v_i) must be 0 for i < 7.
        for i in 0..7 {
            assert!(
                !Gf2p8_11d::from(basis[i]).trace(),
                "Trace of v_{} must be 0 to allow extension",
                i
            );
        }

        // 4. Check Linear Independence
        // We verify that the 8 elements are linearly independent over GF(2)
        // by checking if they can be reduced to an identity matrix (Rank 8).
        assert!(
            is_linearly_independent(&basis),
            "Basis elements are not linearly independent"
        );
    }

    /// Helper to check linear independence over GF(2) using Gaussian Elimination
    fn is_linearly_independent(basis: &[u8]) -> bool {
        let mut matrix = basis.to_vec();
        let mut rank = 0;

        for bit in (0..8).rev() {
            // Find a row with a 1 at the current bit position
            let mut pivot = None;
            for i in rank..matrix.len() {
                if (matrix[i] >> bit) & 1 != 0 {
                    pivot = Some(i);
                    break;
                }
            }

            if let Some(p) = pivot {
                matrix.swap(rank, p);
                for i in 0..matrix.len() {
                    if i != rank && (matrix[i] >> bit) & 1 != 0 {
                        matrix[i] ^= matrix[rank];
                    }
                }
                rank += 1;
            }
        }
        rank == basis.len()
    }
}
