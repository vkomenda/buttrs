use super::bit_matrix::BitMatrix;

pub const FIELD_SIZE: usize = 256;
pub const EXP_TABLE_SIZE: usize = FIELD_SIZE * 2 - 2;

pub trait Gf2p8: Sized + Copy + From<u8> + Into<u8> + PartialEq {
    const POLY: u16;
    const PRIM_ELEM: Self;

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

    fn into_usize(self) -> usize {
        let byte: u8 = self.into();
        byte as usize
    }

    fn exp_log_tables() -> ([u8; EXP_TABLE_SIZE], [u8; FIELD_SIZE]) {
        let mut exp_table = [0u8; EXP_TABLE_SIZE];
        let mut log_table = [0u8; FIELD_SIZE];

        let mut x = 1u8;
        // build exp[0..254], log for non-zero
        for (i, e) in exp_table.iter_mut().take(FIELD_SIZE - 1).enumerate() {
            *e = x;
            log_table[x as usize] = i as u8;

            let gf_x: Self = x.into();
            x = gf_x.mul(Self::PRIM_ELEM).into();
        }

        // copy for overflow-friendly indexing
        for i in 0..FIELD_SIZE - 1 {
            exp_table[FIELD_SIZE - 1 + i] = exp_table[i];
        }

        (exp_table, log_table)
    }

    fn inv_table(
        exp_table: &[u8; EXP_TABLE_SIZE],
        log_table: &[u8; FIELD_SIZE],
    ) -> [u8; FIELD_SIZE] {
        let mut inv_table = [0u8; FIELD_SIZE];

        for (i, e) in inv_table.iter_mut().enumerate().skip(1) {
            let li = log_table[i] as usize;
            *e = exp_table[FIELD_SIZE - 1 - li];
        }

        inv_table
    }

    /// Brute-force the multiplicative inverse lookup table.
    ///
    /// 0 has no inverse. It is preserved in the output, so the case of 0 needs to be covered with checks.
    fn iter_inverses() -> impl Iterator<Item = Self> {
        std::iter::once(0u8.into()).chain((1u8..=255).map(|a| {
            let gf_a: Self = a.into();
            for b in 1u8..=255 {
                let gf_b = b.into();
                if gf_a.mul(gf_b) == 1u8.into() {
                    return gf_b;
                }
            }
            panic!("Cannot compute mul inverse of {a}");
        }))
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

    /// Create a bit matrix for (x * self) mod POLY
    fn into_bit_matrix(self) -> BitMatrix {
        let mut m = BitMatrix([0u8; 8]);
        for i in 0..8 {
            m.0[i] = self.mul((1u8 << i).into()).into();
        }

        m.transpose()
    }
}

pub trait CantorBasis<G: Gf2p8>:
    Sized + Copy + Clone + FromIterator<G> + IntoIterator<Item = G>
{
    fn new() -> Self {
        let mut basis = Vec::new();

        // Start with v0 = 1
        let mut current: G = 1u8.into();
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
        basis.into_iter().collect()
    }

    fn into_fft_twiddle_matrices(self) -> Vec<BitMatrix> {
        // Convert each basis element into an 8x8 bit matrix.
        // In the actual Firedancer assembly, these are the constants
        // that get loaded into ZMM registers for vgf2p8affineqb.
        self.into_iter().map(|a| a.into_bit_matrix()).collect()
    }

    /// Evaluates the erasure locator polynomial E(x) at point alpha_i.
    /// E(x) = product over missing indices j of (x ^ alpha_j).
    fn eval_erasure_locator_poly(&self, i: u8, erased_indices: &[u8]) -> G {
        let alpha_i = self.get_subspace_point(i);
        let mut eval: G = 1u8.into();

        for &j in erased_indices {
            if i == j {
                continue;
            }
            let alpha_j = self.get_subspace_point(j);
            let factor = alpha_i.add(alpha_j);
            eval = eval.mul(factor);
        }
        eval
    }

    /// Returns the i-th point in the basis subspace.
    fn get_subspace_point(&self, i: u8) -> G {
        let mut point: G = 0u8.into();
        for (bit, elem) in self.into_iter().enumerate() {
            if (i >> bit) & 1 != 0 {
                point = point.add(elem);
            }
        }
        point
    }

    fn iter_subspace_points(&self) -> (usize, impl Iterator<Item = G>) {
        let basis_len = self.into_iter().count();
        let num_points = 1 << basis_len;
        (
            num_points,
            (0..num_points).map(|i| self.get_subspace_point(i as u8)),
        )
    }
}

/// Precompted lookup table group operations.
pub trait Gf2p8Lut: Gf2p8 {
    /// Multiplication by table lookup.
    fn mul_lut(self, other: Self) -> Self;

    /// Multiplicative inverse by table lookup.
    fn inv_lut(self) -> Self;
}

/// Precompted lookup table operations on the Cantor basis subspace.
pub trait CantorBasisLut<G: Gf2p8Lut> {
    /// Evaluates the erasure locator polynomial E(x) at point alpha_i.
    /// E(x) = product over missing indices j of (x ^ alpha_j).
    fn eval_erasure_locator_poly_lut(&self, i: u8, erased_indices: &[u8]) -> G {
        let alpha_i = self.get_subspace_point_lut(i);
        let mut eval: G = 1u8.into();

        for &j in erased_indices {
            if i == j {
                continue;
            }
            let alpha_j = self.get_subspace_point_lut(j);
            eval = eval.mul_lut(alpha_i.add(alpha_j));
        }
        eval
    }

    /// Returns the i-th point in the basis subspace.
    fn get_subspace_point_lut(&self, i: u8) -> G;
}
