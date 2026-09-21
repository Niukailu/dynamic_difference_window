use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, path::Path};
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Window {
    pub base: u64,
    pub bits: Vec<i32>,
}
impl Window {
    pub fn new(base: u64, bits: Vec<i32>) -> Result<Self> {
        ensure!(
            bits.len() <= 20
                && bits.windows(2).all(|x| x[0] < x[1])
                && bits.iter().all(|&b| (0..64).contains(&b)),
            "invalid window bits"
        );
        let mask = bits.iter().fold(0u64, |m, &b| m | (1u64 << b));
        Ok(Self {
            base: base & !mask,
            bits,
        })
    }
    pub fn point(base: u64) -> Self {
        Self { base, bits: vec![] }
    }
    pub fn size(&self) -> usize {
        1usize << self.bits.len()
    }
    pub fn mask(&self) -> u64 {
        self.bits.iter().fold(0, |m, &b| m | (1u64 << b))
    }
    pub fn value(&self, index: usize) -> u64 {
        self.bits
            .iter()
            .enumerate()
            .fold(self.base, |v, (i, &b)| v | (((index >> i) & 1) as u64) << b)
    }
    pub fn index(&self, value: u64) -> Option<usize> {
        if value & !self.mask() != self.base {
            return None;
        }
        Some(
            self.bits
                .iter()
                .enumerate()
                .fold(0, |v, (i, &b)| v | (((value >> b) & 1) as usize) << i),
        )
    }
    pub fn shard(&self, rank: usize, devices: usize) -> Result<Self> {
        ensure!(
            devices.is_power_of_two() && rank < devices && self.size() >= devices,
            "window too small for row sharding"
        );
        let count = self.bits.len() - devices.trailing_zeros() as usize;
        let mut base = self.base;
        for (j, &bit) in self.bits[count..].iter().enumerate() {
            base |= (((rank >> j) & 1) as u64) << bit;
        }
        Self::new(base, self.bits[..count].to_vec())
    }
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        if (self.base ^ other.base) & !(self.mask() | other.mask()) != 0 {
            return None;
        }
        Self::new(
            self.base | other.base,
            self.bits
                .iter()
                .copied()
                .filter(|b| other.bits.contains(b))
                .collect(),
        )
        .ok()
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub cipher: String,
    pub mode: String,
    pub word_bits: u32,
    pub left: u64,
    pub right: u64,
    pub width: usize,
    pub rounds: usize,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [16, 24, 32, 48, 64].contains(&self.word_bits),
            "unsupported word size"
        );
        ensure!(
            ["simon", "simeck"].contains(&self.cipher.as_str())
                && ["difference", "linear"].contains(&self.mode.as_str()),
            "invalid cipher/mode"
        );
        ensure!(
            self.rounds > 0 && self.width <= 20 && self.width <= self.word_bits as usize,
            "invalid width/rounds"
        );
        ensure!(
            self.left | self.right != 0,
            "zero input is not a distinguisher"
        );
        if self.word_bits < 64 {
            ensure!(
                (self.left | self.right) < 1u64 << self.word_bits,
                "input outside word size"
            );
        }
        Ok(())
    }
    pub fn initial(&self) -> (Window, Window) {
        if self.mode == "linear" {
            (Window::point(self.right), Window::point(self.left))
        } else {
            (Window::point(self.left), Window::point(self.right))
        }
    }
    pub fn physical(&self, pair: (u64, u64)) -> (u64, u64) {
        if self.mode == "linear" {
            (pair.1, pair.0)
        } else {
            pair
        }
    }
}
pub fn number(text: &str) -> Result<u64> {
    Ok(if let Some(s) = text.strip_prefix("0x") {
        u64::from_str_radix(s, 16)?
    } else {
        text.parse()?
    })
}
pub struct Plan {
    pub config: Config,
    pub windows: Vec<Window>,
    pub records: Vec<Value>,
}
impl Plan {
    pub fn read(path: &Path) -> Result<Self> {
        Self::parse(&fs::read_to_string(path)?)
    }
    pub fn parse(text: &str) -> Result<Self> {
        let records: Vec<Value> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        let mut value = records.first().context("empty plan")?["config"].clone();
        let rounds: Vec<_> = records
            .into_iter()
            .filter(|r| r.get("round").is_some())
            .collect();
        ensure!(!rounds.is_empty(), "no rounds in plan");
        if value.get("rounds").is_none() {
            value["rounds"] = rounds.len().into();
        }
        let config: Config = serde_json::from_value(value)?;
        config.validate()?;
        let mut windows = vec![];
        for (i, record) in rounds.iter().enumerate() {
            ensure!(
                record["round"].as_u64() == Some((i + 1) as u64),
                "noncontiguous rounds"
            );
            let bits: Vec<i32> = serde_json::from_value(record["window_bits"].clone())?;
            ensure!(
                bits.iter().all(|&b| b >= 0 && b < config.word_bits as i32),
                "window bit outside cipher word"
            );
            ensure!(
                bits.len() <= config.width,
                "window exceeds configured width"
            );
            let base = number(
                record["window_base"]
                    .as_str()
                    .context("missing window base")?,
            )?;
            let mask = if config.word_bits == 64 {
                u64::MAX
            } else {
                (1u64 << config.word_bits) - 1
            };
            ensure!(base & !mask == 0, "window base outside cipher word");
            let output = record["output"].as_array().context("missing output pair")?;
            ensure!(output.len() == 2, "output must contain two words");
            for word in output {
                ensure!(
                    number(word.as_str().context("output must be a hex string")?)? & !mask == 0,
                    "output outside cipher word"
                );
            }
            ensure!(
                record["log2_max"].as_f64().is_some_and(f64::is_finite),
                "missing finite peak"
            );
            windows.push(Window::new(base, bits)?);
        }
        ensure!(windows.len() == config.rounds, "incomplete plan");
        Ok(Self {
            config,
            windows,
            records: rounds,
        })
    }
    pub fn endpoint(&self) -> Result<(u64, u64)> {
        let last = self.records.last().context("no final endpoint")?;
        Ok((
            number(last["output"][0].as_str().context("missing output")?)?,
            number(last["output"][1].as_str().context("missing output")?)?,
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_plans() {
        let config = serde_json::json!({"config":{"cipher":"simon","mode":"difference","word_bits":16,"left":0,"right":1,"width":2,"rounds":1}});
        let row = serde_json::json!({"round":1,"window_base":"0x0","window_bits":[1],"output":["0x0","0x0"],"log2_max":-2.0});
        let text = |r: &Value| format!("{config}\n{r}\n");
        assert!(Plan::parse(&text(&row)).is_ok());
        for (key, value) in [
            ("window_base", serde_json::json!("0x10000")),
            ("window_bits", serde_json::json!([0, 1, 2])),
            ("output", serde_json::json!(["0x0"])),
            ("round", serde_json::json!(2)),
            ("log2_max", Value::Null),
        ] {
            let mut invalid = row.clone();
            invalid[key] = value;
            assert!(Plan::parse(&text(&invalid)).is_err(), "accepted {key}");
        }
        assert!(Plan::parse(&format!("{config}\n")).is_err());
    }
    #[test]
    fn windows() {
        let w = Window::new(u64::MAX, vec![0, 31, 63]).unwrap();
        for i in 0..w.size() {
            assert_eq!(w.index(w.value(i)), Some(i));
        }
        assert!(Window::new(0, vec![2, 1]).is_err());
        for rank in 0..4 {
            let s = w.shard(rank, 4).unwrap();
            for i in 0..s.size() {
                assert_eq!(s.value(i), w.value(rank * s.size() + i));
            }
        }
    }
    #[test]
    fn intersection() {
        let a = Window::new(0, vec![0, 2]).unwrap();
        let b = Window::new(1, vec![1, 2]).unwrap();
        let c = a.intersect(&b).unwrap();
        for x in 0..16 {
            assert_eq!(
                c.index(x).is_some(),
                a.index(x).is_some() && b.index(x).is_some()
            );
        }
    }
}
