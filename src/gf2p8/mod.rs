pub mod bit_matrix;
pub mod generic;

pub use generic::Gf2p8;

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

#[cfg(test)]
mod tests {
    use super::*;

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
            let v_curr = basis[i];
            let v_prev = basis[i - 1];

            // v^2 + v
            let lhs = v_curr.mul(v_curr).add(v_curr);
            assert_eq!(lhs, v_prev, "Chain property failed at index {}", i);
        }

        // 3. Check Trace Conditions
        // For the sequence to be extendable, Tr(v_i) must be 0 for i < 7.
        for i in 0..7 {
            assert!(
                !basis[i].trace(),
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
    fn is_linearly_independent(basis: &[Gf2p8_11d]) -> bool {
        let mut matrix = basis.to_vec();
        let mut rank = 0;

        for bit in (0..8).rev() {
            // Find a row with a 1 at the current bit position
            let mut pivot = None;
            for i in rank..matrix.len() {
                if (u8::from(matrix[i]) >> bit) & 1 != 0 {
                    pivot = Some(i);
                    break;
                }
            }

            if let Some(p) = pivot {
                matrix.swap(rank, p);
                for i in 0..matrix.len() {
                    if i != rank && (u8::from(matrix[i]) >> bit) & 1 != 0 {
                        matrix[i] = matrix[i].add(matrix[rank]);
                    }
                }
                rank += 1;
            }
        }
        rank == basis.len()
    }
}
