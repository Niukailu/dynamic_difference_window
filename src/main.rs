use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use ddw::{
    model::{number, Config, Plan},
    run::{self, Options},
};
use std::path::PathBuf;
#[derive(Parser)]
#[command(version, about = "Rust/CUDA dynamic-window cryptanalysis research")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Args)]
struct Hardware {
    #[arg(long, value_delimiter = ',', default_value = "0")]
    devices: Vec<i32>,
    #[arg(long, default_value_t = 28.)]
    memory_gib: f64,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Args)]
struct Selection {
    #[arg(long, default_value_t = 1)]
    candidates: usize,
    #[arg(long,default_value="mass",value_parser=["mass","peak"])]
    objective: String,
}
#[derive(Subcommand)]
enum Command {
    Replay {
        plan: PathBuf,
        #[arg(long)]
        rounds: Option<usize>,
        #[arg(long)]
        width: Option<usize>,
        #[arg(long)]
        extend: bool,
        #[command(flatten)]
        hardware: Hardware,
        #[command(flatten)]
        selection: Selection,
    },
    /// Lossless single-GPU virtual matrix replay or nested expansion.
    Compressed {
        plan: PathBuf,
        #[arg(long)]
        width: Option<usize>,
        #[arg(long)]
        rounds: Option<usize>,
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value_t = 40.)]
        memory_gib: f64,
        #[arg(long)]
        output: PathBuf,
    },
    Search {
        #[arg(long, default_value_t = 32)]
        word_bits: u32,
        #[arg(long,default_value="simon",value_parser=["simon","simeck"])]
        cipher: String,
        #[arg(long,default_value="difference",value_parser=["difference","linear"])]
        mode: String,
        #[arg(long, default_value = "0")]
        left: String,
        #[arg(long, default_value = "1")]
        right: String,
        #[arg(long, default_value_t = 10)]
        width: usize,
        #[arg(long, default_value_t = 23)]
        rounds: usize,
        #[command(flatten)]
        hardware: Hardware,
        #[command(flatten)]
        selection: Selection,
    },
    /// Join a saved half schedule to its reflection, targeting swapped endpoints.
    Mirror {
        plan: PathBuf,
        #[arg(long, default_value_t = 12)]
        half_rounds: usize,
        #[arg(long)]
        reverse: bool,
        #[arg(long)]
        compressed: bool,
        #[command(flatten)]
        hardware: Hardware,
    },
    Refine {
        plan: PathBuf,
        #[arg(long, default_value="local", value_parser=["local","posterior"])]
        strategy: String,
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value_t = 28.)]
        memory_gib: f64,
        #[arg(long, default_value_t = 128.)]
        host_memory_gib: f64,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 2)]
        passes: usize,
        #[arg(long)]
        output: PathBuf,
    },
    /// Exact union with a reflected schedule, for swapped endpoints.
    SymmetricUnion {
        plan: PathBuf,
        #[arg(long)]
        rounds: Option<usize>,
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value_t = 40.)]
        memory_gib: f64,
        #[arg(long)]
        output: PathBuf,
    },
    /// Inclusion-exclusion over up to four path sets with swapped endpoints.
    PathUnion {
        #[arg(required = true)]
        plans: Vec<PathBuf>,
        #[arg(long)]
        reflect: bool,
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value_t = 40.)]
        memory_gib: f64,
        #[arg(long)]
        output: PathBuf,
    },
    Validate {
        file: PathBuf,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long, default_value_t = 1e-10)]
        tolerance: f64,
    },
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Replay {
            plan,
            rounds,
            width,
            extend,
            hardware,
            selection,
        } => {
            let plan = Plan::read(&plan)?;
            let mut config = plan.config.clone();
            if let Some(rounds) = rounds {
                config.rounds = rounds;
            }
            if let Some(width) = width {
                config.width = width;
            }
            let options = Options {
                devices: hardware.devices,
                memory_gib: hardware.memory_gib,
                candidates: selection.candidates,
                objective: selection.objective,
                extend,
                expand: width.is_some(),
            };
            run::execute(&config, Some(&plan), &options, &hardware.output)
        }
        Command::Compressed {
            plan,
            width,
            rounds,
            device,
            memory_gib,
            output,
        } => ddw::compressed::execute(
            &Plan::read(&plan)?,
            width,
            rounds,
            device,
            memory_gib,
            &output,
        ),
        Command::Search {
            word_bits,
            cipher,
            mode,
            left,
            right,
            width,
            rounds,
            hardware,
            selection,
        } => {
            let config = Config {
                word_bits,
                cipher,
                mode,
                left: number(&left)?,
                right: number(&right)?,
                width,
                rounds,
            };
            let options = Options {
                devices: hardware.devices,
                memory_gib: hardware.memory_gib,
                candidates: selection.candidates,
                objective: selection.objective,
                extend: false,
                expand: false,
            };
            run::execute(&config, None, &options, &hardware.output)
        }
        Command::Mirror {
            plan,
            half_rounds,
            reverse,
            compressed,
            hardware,
        } => {
            let plan = Plan::read(&plan)?.mirrored(half_rounds, reverse)?;
            if compressed {
                anyhow::ensure!(
                    hardware.devices.len() == 1,
                    "compressed mirror currently uses one device"
                );
                return ddw::compressed::execute(
                    &plan,
                    None,
                    None,
                    hardware.devices[0],
                    hardware.memory_gib,
                    &hardware.output,
                );
            }
            let options = Options {
                devices: hardware.devices,
                memory_gib: hardware.memory_gib,
                candidates: 1,
                objective: "mass".into(),
                extend: false,
                expand: false,
            };
            run::execute(&plan.config, Some(&plan), &options, &hardware.output)
        }
        Command::Refine {
            plan,
            strategy,
            device,
            memory_gib,
            host_memory_gib,
            candidates,
            passes,
            output,
        } => {
            if strategy == "posterior" {
                ddw::posterior::execute(
                    &Plan::read(&plan)?,
                    device,
                    memory_gib,
                    host_memory_gib,
                    passes,
                    &output,
                )
            } else {
                ddw::refine::execute(
                    &Plan::read(&plan)?,
                    device,
                    memory_gib,
                    host_memory_gib,
                    candidates,
                    passes,
                    &output,
                )
            }
        }
        Command::SymmetricUnion {
            plan,
            rounds,
            device,
            memory_gib,
            output,
        } => ddw::union::execute(&plan, rounds, device, memory_gib, &output),
        Command::PathUnion {
            plans,
            reflect,
            device,
            memory_gib,
            output,
        } => ddw::union::multiple(&plans, reflect, device, memory_gib, &output),
        Command::Validate {
            file,
            reference,
            tolerance,
        } => {
            println!("{}", run::validate(&file, reference.as_deref(), tolerance)?);
            Ok(())
        }
    }
}
