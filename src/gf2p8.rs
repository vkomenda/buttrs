use crate::bit_matrix::BitMatrix;

pub trait Gf2p8: Sized + Copy + From<u8> + Into<u8> + PartialEq + Eq {
    const POLY: u16;

    fn add(self, other: Self) -> Self {
        Self::from(self.into() ^ other.into())
    }

    fn mul(self, other: Self) -> Self {
        let mut a = self.into() as u16;
        let mut b = other.into() as u16;
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
        (res as u8).into()
    }

    /// Trace function for GF(2^8) over GF(2)
    /// Tr(x) = x + x^2 + x^4 + x^8 + x^16 + x^32 + x^64 + x^128
    fn trace(self) -> bool {
        let mut t = self;
        let mut sum: u8 = self.into();
        for _ in 0..7 {
            t = t.mul(t); // squaring
            sum ^= t.into();
        }
        (sum & 1) != 0
    }

    /// Solves the quadratic equation x^2 + x = c
    /// Returns one of the two solutions (x and x+1)
    fn solve_quadratic(self) -> Option<Self> {
        if self.trace() {
            return None;
        }
        // For GF(2^8), we can simply brute force or use the Half-Trace.
        // Brute force is fine for a one-time basis generation.
        for i in 0u8..=255 {
            let x: Self = i.into();
            if x.mul(x).add(x) == self {
                return Some(x);
            }
        }
        None
    }

    fn generate_cantor_basis() -> Vec<Self> {
        let mut basis = Vec::new();

        // Start with v0 = 1
        let mut current: Self = 1u8.into();
        basis.push(current);

        // Try to extend the chain using v_i^2 + v_i = v_{i-1}
        while let Some(next) = current.solve_quadratic() {
            // We have two solutions: 'next' and 'next + 1'.
            // We must pick the one with Trace 0 to ensure the next level exists.
            if !next.trace() {
                current = next;
            } else {
                current = next.add(1u8.into());
            }
            basis.push(current);

            // Stop if we hit 8 elements (full field) or can't solve anymore
            if basis.len() == 8 {
                break;
            }
        }
        basis
    }

    /// Create a bit-matrix for (x * self) mod POLY
    fn into_bit_matrix(self) -> BitMatrix {
        let mut cols = [0u8; 8];
        for i in 0..8 {
            cols[i] = self.mul((1u8 << i).into()).into();
        }
        // To match vgf2p8affineqb, we store it so apply() does bit-matrix multiplication
        BitMatrix(cols)
    }

    fn get_fft_twiddle_matrices() -> Vec<BitMatrix> {
        let basis = Self::generate_cantor_basis();

        // Convert each basis element into an 8x8 bit-matrix.
        // In the actual Firedancer assembly, these are the constants
        // that get loaded into ZMM registers for vgf2p8affineqb.
        basis.into_iter().map(Self::into_bit_matrix).collect()
    }
}
