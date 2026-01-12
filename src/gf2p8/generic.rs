use super::bit_matrix::BitMatrix;

pub trait Gf2p8: Sized + Copy + From<u8> + Into<u8> + PartialEq {
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

    /// Multiplicative inverse
    fn inv(self) -> Self {
        todo!()
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

    /// Solves the quadratic equation x^2 + x = self
    /// Returns one of the two solutions (x and x+1)
    fn solve_quadratic(self) -> Option<Self> {
        if self.trace() {
            return None;
        }
        // For GF(2^8), we can simply brute force or use the half trace.
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

    /// Create a bit matrix for (x * self) mod POLY
    fn into_bit_matrix(self) -> BitMatrix {
        let mut m = BitMatrix([0u8; 8]);
        for i in 0..8 {
            m.0[i] = self.mul((1u8 << i).into()).into();
        }

        m.transpose()
    }

    fn get_fft_twiddle_matrices() -> Vec<BitMatrix> {
        let basis = Self::generate_cantor_basis();

        // Convert each basis element into an 8x8 bit matrix.
        // In the actual Firedancer assembly, these are the constants
        // that get loaded into ZMM registers for vgf2p8affineqb.
        basis.into_iter().map(Self::into_bit_matrix).collect()
    }

    /// Evaluates the erasure locator polynomial E(x) at point alpha_i.
    /// E(x) = product over missing indices j of (x ^ alpha_j).
    fn eval_erasure_locator_poly(i: u8, erased_indices: &[u8], basis: &[Self]) -> Self {
        let alpha_i = Self::get_subspace_point(i, basis);
        let mut eval: Self = 1u8.into();

        for &j in erased_indices {
            if i == j {
                continue;
            }
            let alpha_j = Self::get_subspace_point(j, basis);
            let factor = alpha_i.add(alpha_j);
            eval = eval.mul(factor);
        }
        eval
    }

    /// Returns the i-th point in the basis subspace.
    fn get_subspace_point(index: u8, basis: &[Self]) -> Self {
        let mut point: Self = 0u8.into();
        for bit in 0..6 {
            if (index >> bit) & 1 != 0 {
                point = point.add(basis[bit]);
            }
        }
        point
    }
}
