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
    Sized + Copy + Clone + FromIterator<G> + IntoIterator<Item = G> + AsRef<[G]>
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
        // Reverse the order of indices to match the reversed storage order.
        for (bit, elem) in (0..8).rev().zip(self.into_iter()) {
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

    /// Generates the LCH Twiddle Tower for an N-point FFT.
    fn generate_lch_twiddle_tower<const N: usize>(&self) -> Vec<BitMatrix> {
        let k = N.trailing_zeros() as usize;

        let mut current_basis: Vec<G> = self.into_iter().take(k).collect();
        let mut twiddles = Vec::with_capacity(k);

        // Build the tower from the bottom
        for i in (0..k).rev() {
            // The pivot for the current butterfly layer
            let beta = current_basis[i];
            twiddles.push(beta.into_bit_matrix());

            // Transform the lower-level basis elements for the sub-FFTs.
            // This is the "Subspace Polynomial" step: L(x) = x^2 + x*beta.
            // It ensures that the sub-problems see unique points.
            for j in 0..i {
                let x = current_basis[j];
                current_basis[j] = (x.mul(x)).add(x.mul(beta));
            }
        }

        twiddles.reverse();
        twiddles
    }

    fn eval_subspace_poly(&self, k: u8, x: G) -> G {
        let mut val: G = 1.into();
        for a in self.span_by_gray_code(k) {
            let sum = x.add(a);
            val = val.mul(sum);
        }
        val
    }

    fn chain_of_subspaces(&self) -> Vec<Vec<G>> {
        (0..9).map(|k| self.span(k)).collect()
    }

    fn span(&self, k: u8) -> Vec<G> {
        let size = 1 << k;
        let mut res = Vec::with_capacity(size);
        for i in 0..size {
            let mut sum: G = 0.into();
            for (j, v) in self.into_iter().take(k as usize).enumerate() {
                if (i >> j) & 1 == 1 {
                    sum = sum.add(v);
                }
            }
            res.push(sum);
        }
        res
    }

    fn span_by_gray_code(&self, k: u8) -> Vec<G> {
        let size = 1 << k;
        let mut span: Vec<G> = vec![0u8.into(); size];
        for i in 1..size {
            let lsb = i.trailing_zeros() as usize;
            span[i] = span[i ^ (1 << lsb)].add(self.as_ref()[lsb]);
        }
        span
    }

    // FIXME: Broken!
    //
    // /// Generates the compact LUT for the subspace polynomial s_k(x).
    // /// The const K is the table size 2^(8-k). the level index k is derived as 8 - log2(K).
    // fn gen_compact_subspace_poly_lut<const K: usize>(&self) -> [G; K] {
    //     let log2_k = K.trailing_zeros() as usize;
    //     let k = 8 - log2_k;

    //     let mut table = [0u8.into(); K];
    //     let basis = self.as_ref();

    //     // Pre-compute projections b_l = s_l(v_l) for l < k.
    //     // These are the basis point projections required to evaluate s_k.
    //     let mut basis_image = [G::from(0u8); 8];
    //     let mut w = basis.to_vec();
    //     for l in 0..k {
    //         let b_l = w[l];
    //         basis_image[l] = b_l;
    //         for w_j in w.iter_mut().take(k).skip(l + 1) {
    //             // s_{l+1}(v_j) = s_l(v_j) * (s_l(v_j) + s_l(v_l))
    //             *w_j = w_j.mul(w_j.add(b_l));
    //         }
    //     }

    //     // Compute s_k(X_{r << k}) for all r in the compressed range.
    //     for r in 0..K {
    //         // We evaluate s_k only at coset representatives of the subspace V_k.
    //         // These are points where the lower k bits are zero.
    //         let mut val = self.get_subspace_point((r << k) as u8);

    //         // Apply the chain of subspace polynomials s_0 -> s_1 -> ... -> s_k
    //         for &b_l in basis_image.iter().take(k) {
    //             val = val.mul(val.add(b_l));
    //         }
    //         table[r] = val;
    //     }

    //     table
    // }

    /// Generates a LUT for the subspace polynomial s_k(x).
    /// The table index is the field element (as u8), and the value is s_k(index).
    /// s_0(x) = x
    /// s_{j+1}(x) = s_j(x) * (s_j(x) + s_j(v_j))
    fn gen_subspace_poly_lut(&self, k: usize) -> [G; FIELD_SIZE] {
        let mut table = [0u8.into(); FIELD_SIZE];

        // Base case: s_0(x) = x
        for x in 0..FIELD_SIZE {
            table[x] = G::from(x as u8);
        }

        let basis = self.as_ref();
        // We only need to iterate if k > 0
        for v_j in basis.iter().take(k) {
            // b_j = s_j(v_j)
            // To find this, we evaluate the current state of s_j at point v_j.
            // Since our table currently holds s_j(x) for all x,
            // we just look up the index corresponding to basis element v_j.
            let b_j = table[(*v_j).into_usize()];

            // Update the table: s_{j+1}(x) = s_j(x) * (s_j(x) + s_j(v_j))
            for x in table.iter_mut() {
                let s_j_x = *x;
                *x = s_j_x.mul(s_j_x.add(b_j));
            }
        }

        table
    }

    /// Generates all subspace polynomial LUTs from s_0 to s_8.
    /// Returns an array where [j][x] contains s_j(x).
    /// s_0(x) = x
    /// s_{j+1}(x) = s_j(x) * (s_j(x) + s_j(v_j))
    fn gen_all_subspace_poly_luts(&self) -> [[G; FIELD_SIZE]; 9] {
        let mut luts = [[0u8.into(); FIELD_SIZE]; 9];

        // 1. Initialize s_0(x) = x
        for (x, s_0_x) in luts[0].iter_mut().enumerate() {
            *s_0_x = G::from(x as u8);
        }

        let basis = self.as_ref();

        // 2. Iteratively compute s_{j+1} from s_j
        for j in 0..8 {
            // b_j is the basis projection: s_j(v_j)
            // We look up the basis element v_j in the current s_j table
            let b_j = luts[j][basis[j].into_usize()];

            for x in 0..FIELD_SIZE {
                let s_j_x = luts[j][x];
                // s_{j+1}(x) = s_j(x) * (s_j(x) + b_j)
                luts[j + 1][x] = s_j_x.mul(s_j_x.add(b_j));
            }
        }

        luts
    }

    /// Generates 1/p_i values, for 0 <= i < 256.
    fn gen_normalization_factors(
        &self,
        subspace_poly_luts: &[[G; FIELD_SIZE]; 9],
        inv_lut: &[u8; FIELD_SIZE],
    ) -> [u8; FIELD_SIZE] {
        let basis = self.as_ref();
        let basis_image: Vec<G> = (0..8)
            .map(|i| subspace_poly_luts[i][basis[i].into_usize()])
            .collect();
        let mut factors = [0u8; FIELD_SIZE];

        for (i, f) in factors.iter_mut().enumerate() {
            let mut p: G = 1u8.into();
            for (j, &b) in basis_image.iter().enumerate() {
                if (i >> j) & 1 == 1 {
                    p = p.mul(b);
                }
            }
            *f = inv_lut[p.into_usize()];
        }

        factors
    }
}

/// Precomputed lookup table group operations.
pub trait Gf2p8Lut: Gf2p8 {
    /// Multiplication by table lookup.
    fn mul_lut(self, other: Self) -> Self;

    /// Multiplicative inverse by table lookup.
    fn inv_lut(self) -> Self;

    /// Helper to multiply a shard by a scalar
    fn scale_shard(self, shard: &mut [u8]) {
        if self == 1u8.into() {
            return;
        }
        if self == 0u8.into() {
            shard.fill(0);
            return;
        }

        for byte in shard.iter_mut() {
            *byte = self.mul_lut(Self::from(*byte)).into();
        }
    }
}

/// Precompted lookup table operations on the Cantor basis subspace.
pub trait CantorBasisLut<G: Gf2p8Lut> {
    fn get_basis_point_lut(&self, i: u8) -> G;

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

    /// Returns the i-th point $s_i(v_i)$ in the basis subspace.
    fn get_subspace_point_lut(&self, i: u8) -> G;

    fn eval_subspace_poly_lut(&self, k: u8, x: G) -> G;

    /// A basis of the algebraic ring $F_{2^m}[x]/(x^{2^m}-x)$ which forms the evaluation space.
    fn compute_evaluation_space_basis_point(&self, i: u8, x: G) -> G {
        let m = 8;
        let mut result: G = 1u8.into();
        let mut s_j_x = x;

        for j in 0..m {
            let b_j = self.eval_subspace_poly_lut(j, x);

            if (i >> j) & 1 == 1 {
                let term = s_j_x.mul_lut(b_j.inv_lut());
                result = result.mul_lut(term);
            }

            // Update s_j_x to s_{j+1}_x by recursion:
            // s_{j+1}(x) = s_j(x) * (s_j(x) + s_j(v_j))
            s_j_x = s_j_x.mul_lut(s_j_x.add(b_j));
        }

        result
    }
}

/// The Lin-Chung-Han basis
pub trait LchBasisLut<G: Gf2p8Lut>: CantorBasisLut<G> {
    /// Evaluate the i-th LCH basis polynomial at point x.  The default implementation assumes a
    /// Cantor basis in the evaluation domain, which doesn't require scaling terms by a
    /// normalization factor.
    fn eval_lch_basis_poly(&self, i: u8, x: G) -> G {
        let mut result: G = 1u8.into();

        for j in 0u8..8 {
            if (i >> j) & 1 == 1 {
                let s_j_x = self.eval_subspace_poly_lut(j, x);
                result = result.mul(s_j_x);
            }
        }

        result
    }

    /// $\overline{D}_h (x)$
    fn eval_transform_domain_poly(&self, coeffs: &[G], x: G) -> G {
        // debug_assert!((0..=8).any(|k| coeffs.len() == 2usize.pow(k) - 1));
        // let k = coeffs.len().trailing_zeros() as usize;
        let mut result: G = 0u8.into();

        for (i, d) in coeffs.iter().enumerate() {
            let term = d.mul(self.eval_lch_basis_poly(i as u8, x));
            result = result.add(term);
        }

        result
    }
}

pub trait Fft<G: Gf2p8Lut>: CantorBasisLut<G> + LchBasisLut<G> {
    /// Algorithm 1 in LCH paper.
    fn fft(&self, coeffs: &mut [G], k: u8, beta: G) {
        if k == 0 {
            return;
        }

        let half = 1 << (k - 1);

        // Fetch the twiddle factor T = s_{k-1}(beta)
        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);

        // Butterfly stage (line 3-6)
        for i in 0..half {
            let d_i = coeffs[i];
            let d_i_half = coeffs[i + half];

            // g_i_0 = d_i + T * d_{i+half}
            let g_i_0 = d_i.add(twiddle.mul(d_i_half));

            // g_i_1 = g_i_0 + d_{i+half}
            // This is equivalent to d_i + (T + 1) * d_{i+half}
            let g_i_1 = g_i_0.add(d_i_half);

            coeffs[i] = g_i_0;
            coeffs[i + half] = g_i_1;
        }

        // Recursive calls (line 7-8)
        // Left branch: FFT(g_0, k-1, beta)
        self.fft(&mut coeffs[..half], k - 1, beta);

        // Right branch: FFT(g_1, k-1, v_{k-1} + beta)
        let next_beta = beta.add(self.get_basis_point_lut(k - 1));
        self.fft(&mut coeffs[half..], k - 1, next_beta);
    }

    /// Algorithm 2 in LCH paper.
    fn ifft(&self, evals: &mut [G], k: u8, beta: G) {
        if k == 0 {
            return;
        }

        let half = 1 << (k - 1);

        self.ifft(&mut evals[..half], k - 1, beta);

        let next_beta = beta.add(self.get_basis_point_lut(k - 1));
        self.ifft(&mut evals[half..], k - 1, next_beta);

        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);

        for i in 0..half {
            let g_i_0 = evals[i];
            let g_i_1 = evals[i + half];

            let d_i_half = g_i_0.add(g_i_1);
            let d_i = g_i_0.add(twiddle.mul(d_i_half));

            evals[i] = d_i;
            evals[i + half] = d_i_half;
        }
    }

    fn fft_sharded(&self, shards: &mut [&mut [u8]], k: u8, beta: G) {
        if k == 0 {
            return;
        }
        let half = 1 << (k - 1);

        let (left, right) = shards.split_at_mut(half);
        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);

        for i in 0..half {
            let (s_i, s_i_half) = (&mut left[i], &mut right[i]);
            for (byte_l, byte_r) in s_i.iter_mut().zip(s_i_half.iter_mut()) {
                let d_i = G::from(*byte_l);
                let d_i_half = G::from(*byte_r);
                let g_i_0 = d_i.add(twiddle.mul(d_i_half));
                let g_i_1 = g_i_0.add(d_i_half);
                *byte_l = g_i_0.into();
                *byte_r = g_i_1.into();
            }
        }

        self.fft_sharded(left, k - 1, beta);
        self.fft_sharded(right, k - 1, beta.add(self.get_basis_point_lut(k - 1)));
    }

    fn ifft_sharded(&self, shards: &mut [&mut [u8]], k: u8, beta: G) {
        if k == 0 {
            return;
        }
        let half = 1 << (k - 1);

        let (left, right) = shards.split_at_mut(half);
        self.ifft_sharded(left, k - 1, beta);
        self.ifft_sharded(right, k - 1, beta.add(self.get_basis_point_lut(k - 1)));

        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);

        for i in 0..half {
            let (s_i, s_i_half) = (&mut left[i], &mut right[i]);
            for (byte_l, byte_r) in s_i.iter_mut().zip(s_i_half.iter_mut()) {
                let g_i_0 = G::from(*byte_l);
                let g_i_1 = G::from(*byte_r);
                let d_i_half = g_i_0.add(g_i_1);
                let d_i = g_i_0.add(twiddle.mul(d_i_half));
                *byte_l = d_i.into();
                *byte_r = d_i_half.into();
            }
        }
    }

    /// Systematic Reed-Solomon encoding: The message stays the same. Only parity is modified.
    ///
    /// This method is for the special case when the number of message shards equals the number of
    /// parity shards, and is a power of 2.
    fn encode_systematic(&self, message_shards: &[&[u8]], parity_shards: &mut [&mut [u8]]) {
        let num_parity = parity_shards.len();
        let log_num_parity = num_parity.trailing_zeros() as u8;

        // TODO: this can be done by the caller, in which case only the mut buffer is required.
        for (m_shard, p_shard) in message_shards.iter().zip(parity_shards.iter_mut()) {
            p_shard.copy_from_slice(m_shard);
        }

        let beta = self.get_subspace_point_lut(num_parity as u8);
        self.ifft_sharded(parity_shards, log_num_parity, beta);
        self.fft_sharded(parity_shards, log_num_parity, 0u8.into());
    }

    /// IFFT that reads from `msg` and XORs into `parity` (Accumulative IFFT).
    // TODO: test
    fn ifft_accumulate(&self, msg: &[&[u8]], parity: &mut [&mut [u8]], k: u8, beta: G) {
        if k == 0 {
            // Base case: XOR the message shard into the parity shard.
            if let Some(m_shard) = msg.first() {
                for (p_b, m_b) in parity[0].iter_mut().zip(m_shard.iter()) {
                    *p_b ^= *m_b;
                }
            }
            return;
        }

        let half = 1 << (k - 1);
        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);

        let (m_left, m_right) = if msg.len() > half {
            msg.split_at(half)
        } else {
            (msg, &[][..])
        };
        let (p_left, p_right) = parity.split_at_mut(half);

        self.ifft_accumulate(m_left, p_left, k - 1, beta);
        let next_beta = beta.add(self.get_basis_point_lut(k - 1));
        self.ifft_accumulate(m_right, p_right, k - 1, next_beta);

        for i in 0..half {
            let (p_i, p_i_half) = (&mut p_left[i], &mut p_right[i]);
            for (byte_l, byte_r) in p_i.iter_mut().zip(p_i_half.iter_mut()) {
                let g_0 = G::from(*byte_l);
                let g_1 = G::from(*byte_r);

                let d_i_half = g_0.add(g_1);
                let d_i = g_0.add(twiddle.mul(d_i_half));

                *byte_l = d_i.into();
                *byte_r = d_i_half.into();
            }
        }
    }

    /// Syndrome computation. A version for the special case of equal number of message and parity
    /// shards.
    fn compute_syndrome(&self, received_shards: &[&[u8]], syndrome_shards: &mut [&mut [u8]]) {
        let num_parity = syndrome_shards.len();
        let log_num_parity = num_parity.trailing_zeros() as u8;

        let (left, right) = received_shards.split_at(num_parity);

        for (src, dst) in left.iter().zip(syndrome_shards.iter_mut()) {
            dst.copy_from_slice(src);
        }

        self.ifft_sharded(syndrome_shards, log_num_parity, 0u8.into());
        let basis_pt = self.get_basis_point_lut(log_num_parity);
        self.ifft_accumulate(right, syndrome_shards, log_num_parity, basis_pt);
    }
}
