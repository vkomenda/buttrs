#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GF256(pub u8);

impl GF256 {
    pub const POLY: u16 = 0x11D; // x^8 + x^4 + x^3 + x^2 + 1

    pub fn add(self, other: Self) -> Self {
        GF256(self.0 ^ other.0)
    }

    pub fn mul(self, other: Self) -> Self {
        let mut a = self.0 as u16;
        let mut b = other.0 as u16;
        let mut res = 0u16;
        for _ in 0..8 {
            if b & 1 != 0 {
                res ^= a;
            }
            a <<= 1;
            if a & 0x100 != 0 {
                a ^= Self::POLY;
            }
            b >>= 1;
        }
        GF256(res as u8)
    }

    /// Trace function for GF(2^8) over GF(2)
    /// Tr(x) = x + x^2 + x^4 + x^8 + x^16 + x^32 + x^64 + x^128
    pub fn trace(self) -> bool {
        let mut t = self;
        let mut sum = self.0;
        for _ in 0..7 {
            t = t.mul(t); // squaring
            sum ^= t.0;
        }
        (sum & 1) != 0
    }

    /// Solves the quadratic equation x^2 + x = c
    /// Returns one of the two solutions (x and x+1)
    pub fn solve_quadratic(c: Self) -> Option<Self> {
        if c.trace() {
            return None;
        }
        // For GF(2^8), we can simply brute force or use the Half-Trace.
        // Brute force is fine for a one-time basis generation.
        for i in 0..=255 {
            let x = GF256(i);
            if x.mul(x).add(x) == c {
                return Some(x);
            }
        }
        None
    }
}

pub fn generate_cantor_basis() -> Vec<u8> {
    let mut basis = Vec::new();

    // Start with v0 = 1
    let mut current = GF256(1);
    basis.push(current.0);

    // Try to extend the chain using v_i^2 + v_i = v_{i-1}
    while let Some(next) = GF256::solve_quadratic(current) {
        // We have two solutions: 'next' and 'next + 1'.
        // We must pick the one with Trace 0 to ensure the next level exists.
        if !next.trace() {
            current = next;
        } else {
            current = next.add(GF256(1));
        }
        basis.push(current.0);

        // Stop if we hit 8 elements (full field) or can't solve anymore
        if basis.len() == 8 {
            break;
        }
    }
    basis
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitMatrix(pub [u8; 8]);

impl BitMatrix {
    /// Create a bit-matrix for (x * constant) mod 0x11D
    pub fn new_multiplier(constant: u8) -> Self {
        let mut cols = [0u8; 8];
        for i in 0..8 {
            cols[i] = gf256_mul_0x11d(constant, 1 << i);
        }
        // To match vgf2p8affineqb, we store it so apply() does bit-matrix multiplication
        Self(cols)
    }

    /// Software emulation of vgf2p8affineqb (b=0)
    /// This performs: result_vector = Matrix * input_vector over GF(2)
    pub fn apply(&self, val: u8) -> u8 {
        let mut res = 0u8;
        for i in 0..8 {
            if (val >> i) & 1 != 0 {
                res ^= self.0[i]; // XOR the i-th column
            }
        }
        res
    }
}

pub fn get_fft_twiddle_matrices() -> Vec<BitMatrix> {
    let basis = generate_cantor_basis();

    // Convert each basis element into an 8x8 bit-matrix.
    // In the actual Firedancer assembly, these are the constants
    // that get loaded into ZMM registers for vgf2p8affineqb.
    basis.into_iter().map(BitMatrix::new_multiplier).collect()
}

fn gf256_mul_0x11d(mut a: u8, mut b: u8) -> u8 {
    let mut res = 0u16;
    let mut a16 = a as u16;
    for _ in 0..8 {
        if b & 1 != 0 {
            res ^= a16;
        }
        a16 <<= 1;
        if a16 & 0x100 != 0 {
            a16 ^= 0x11D;
        }
        b >>= 1;
    }
    res as u8
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
            BitMatrix::new_multiplier(0x01),
            BitMatrix::new_multiplier(0x02),
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
        assert!(!GF256(1).trace());
    }

    #[test]
    fn test_cantor_basis_properties() {
        let basis = generate_cantor_basis();

        // 1. Check length: For GF(2^8), we expect a basis of 8 elements.
        assert_eq!(basis.len(), 8, "Basis should have 8 elements for GF(2^8)");

        // 2. Check the Chain Property: v_i^2 + v_i = v_{i-1}
        for i in 1..basis.len() {
            let v_curr = GF256(basis[i]);
            let v_prev = GF256(basis[i - 1]);

            // v^2 + v
            let lhs = v_curr.mul(v_curr).add(v_curr);
            assert_eq!(lhs, v_prev, "Chain property failed at index {}", i);
        }

        // 3. Check Trace Conditions
        // For the sequence to be extendable, Tr(v_i) must be 0 for i < 7.
        for i in 0..7 {
            assert!(
                !GF256(basis[i]).trace(),
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
