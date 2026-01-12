#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitMatrix(pub [u8; 8]);

impl BitMatrix {
    /// Software emulation of vgf2p8affineqb (b=0).
    /// This performs: vector_out = self · vector_in, over GF(2).
    /// Matches vgf2p8affineqb: Each u8 in the array is one row of the 8x8 matrix.
    pub fn apply(&self, v: u8) -> u8 {
        let mut res = 0u8;
        for i in 0..8 {
            let prod = v & self.0[i];
            if prod.count_ones().is_multiple_of(2) {
                res |= 1 << i;
            }
        }
        res
    }

    /// Transpose columns to rows for vgf2p8affineqb.
    pub fn transpose(&self) -> Self {
        let mut rows = [0u8; 8];
        for r in 0..8 {
            let mut row_val = 0u8;
            for c in 0..8 {
                let bit = (self.0[c] >> r) & 1;
                row_val |= bit << c;
            }
            rows[r] = row_val;
        }

        Self(rows)
    }
}
