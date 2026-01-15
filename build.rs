#[path = "src/gf2p8/mod.rs"]
mod g;

use g::{
    CantorBasis, CantorBasis11d, Gf2p8, Gf2p8_11d,
    generic::{EXP_TABLE_SIZE, FIELD_SIZE},
};
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn write_points<G>(f: &mut File, it: impl Iterator<Item = G>)
where
    u8: From<G>,
{
    for (i, point) in it.enumerate() {
        if i % 16 == 0 {
            write!(f, "\n    ").unwrap();
        }
        write!(f, "0x{:02x}, ", u8::from(point)).unwrap();
    }
    writeln!(f, "\n];").unwrap();
}

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("tables_11d.rs");
    let mut f = File::create(&dest_path).unwrap();

    writeln!(f, "use crate::BitMatrix;").unwrap();

    let (exp_table, log_table) = Gf2p8_11d::exp_log_tables();
    let inv_table = Gf2p8_11d::inv_table(&exp_table, &log_table);
    write!(f, "\npub const EXP_TABLE: [u8; {}] = [", EXP_TABLE_SIZE).unwrap();
    write_points(&mut f, exp_table.into_iter());
    write!(f, "\npub const LOG_TABLE: [u8; {}] = [", FIELD_SIZE).unwrap();
    write_points(&mut f, log_table.into_iter());
    write!(f, "\npub const INV_TABLE: [u8; {}] = [", FIELD_SIZE).unwrap();
    write_points(&mut f, inv_table.into_iter());

    let basis = CantorBasis11d::new();
    let twiddles = basis.into_fft_twiddle_matrices();

    writeln!(
        f,
        "\npub const TWIDDLE_FACTORS: [BitMatrix; {}] = [",
        twiddles.len()
    )
    .unwrap();
    for mat in twiddles {
        write!(f, "    BitMatrix([").unwrap();
        for byte in mat.0 {
            write!(f, "0x{:02x}, ", byte).unwrap();
        }
        writeln!(f, "]),").unwrap();
    }
    writeln!(f, "];").unwrap();

    let (num_points, points_iter) = basis.iter_subspace_points();

    write!(f, "\npub const CANTOR_SUBSPACE: [u8; {}] = [", num_points).unwrap();
    write_points(&mut f, points_iter);

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/mod.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/generic.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/bit_matrix.rs");
}
