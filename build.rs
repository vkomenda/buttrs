#[path = "src/gf2p8/mod.rs"]
mod g;

use g::{Gf2p8, Gf2p8_11d};
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("twiddle_factors_11d.rs");
    let mut f = File::create(&dest_path).unwrap();

    writeln!(f, "use crate::BitMatrix;").unwrap();

    // Generate the twiddle matrices using your trait logic
    let matrices = Gf2p8_11d::get_fft_twiddle_matrices();

    writeln!(
        f,
        "pub const TWIDDLE_FACTORS_11D: [BitMatrix; {}] = [",
        matrices.len()
    )
    .unwrap();
    for mat in matrices {
        // Format the [u8; 8] as hex for the constant
        write!(f, "    BitMatrix([").unwrap();
        for byte in mat.0 {
            write!(f, "0x{:02x}, ", byte).unwrap();
        }
        writeln!(f, "]),").unwrap();
    }
    writeln!(f, "];").unwrap();

    // Tell Cargo to rerun if field.rs changes
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/gf2p8.rs");
    println!("cargo:rerun-if-changed=src/bit_matrix.rs");
}
