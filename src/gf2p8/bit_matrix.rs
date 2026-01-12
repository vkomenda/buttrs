#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitMatrix(pub [u8; 8]);

impl BitMatrix {
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
