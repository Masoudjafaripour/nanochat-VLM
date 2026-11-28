//! Pretraining data preparation: FineWeb-style parquet shards -> pre-tokenized `.bin` shards.
//!
//! Mirrors the Python pipeline (`nanochat_vlm/dataset.py` + `dataloader.py`):
//! * parquet files sorted by name; the **last** file is `val`, the rest are `train`
//! * documents in file -> row-group -> row order, column `text`
//! * each document is tokenized with `<|bos|>` prepended
//!
//! Tokenization runs in parallel (rayon) per row group; output order is deterministic.
//!
//! `.bin` format (little-endian): a 32-byte header
//! `b"NCVTOK01" | u32 token_bytes (2|4) | u32 reserved | u64 num_tokens | u64 num_docs`
//! followed by `num_tokens` tokens (u16 when the vocab fits, else u32).
//! A `meta.json` next to the shards records the tokenizer, counts and shard list.

use anyhow::Context;
use arrow_array::{Array, LargeStringArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use rayon::prelude::*;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use vlm_tokenizer::Tokenizer;

pub const MAGIC: &[u8; 8] = b"NCVTOK01";
pub const HEADER_BYTES: usize = 32;

/// `dataset.list_parquet_files`: sorted `*.parquet`, skipping partial downloads.
pub fn list_parquet_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "parquet"))
        .collect();
    v.sort();
    Ok(v)
}

/// Read the `text` column of one parquet file, one Vec per row group (in order).
pub fn read_text_row_groups(path: &Path) -> anyhow::Result<Vec<Vec<String>>> {
    let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.parquet_schema();
    let col = (0..schema.num_columns())
        .find(|&i| schema.column(i).name() == "text")
        .context("parquet file has no 'text' column")?;
    let mask = ProjectionMask::leaves(schema, [col]);
    let n_rg = builder.metadata().num_row_groups();
    let mut out = Vec::with_capacity(n_rg);
    for rg in 0..n_rg {
        let file = std::fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_projection(mask.clone())
            .with_row_groups(vec![rg])
            .with_batch_size(8192)
            .build()?;
        let mut texts = Vec::new();
        for batch in reader {
            let col = batch?.column(0).clone();
            if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
                texts.extend((0..a.len()).map(|i| a.value(i).to_string()));
            } else if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
                texts.extend((0..a.len()).map(|i| a.value(i).to_string()));
            } else {
                anyhow::bail!("'text' column is not a string column");
            }
        }
        out.push(texts);
    }
    Ok(out)
}

/// Tokenize documents in parallel, each with BOS prepended; output keeps input order.
pub fn tokenize_docs(tok: &Tokenizer, docs: &[String]) -> Vec<u32> {
    let bos = tok.bos_id();
    let per_doc: Vec<Vec<u32>> = docs
        .par_iter()
        .map(|d| {
            let mut v = vec![bos];
            v.extend(tok.encode(d));
            v
        })
        .collect();
    per_doc.concat()
}

/// Streams tokens into fixed-size `.bin` shards.
pub struct ShardWriter {
    dir: PathBuf,
    prefix: String,
    shard_tokens: usize,
    wide: bool,
    buf: Vec<u32>,
    docs_in_buf: u64,
    pub shards: Vec<(String, u64)>,
    pub total_tokens: u64,
}

impl ShardWriter {
    pub fn new(dir: &Path, prefix: &str, shard_tokens: usize, vocab_size: usize) -> Self {
        Self {
            dir: dir.to_path_buf(),
            prefix: prefix.into(),
            shard_tokens,
            wide: vocab_size > 65536,
            buf: Vec::new(),
            docs_in_buf: 0,
            shards: vec![],
            total_tokens: 0,
        }
    }

    pub fn push(&mut self, tokens: &[u32], docs: u64) -> anyhow::Result<()> {
        self.buf.extend_from_slice(tokens);
        self.docs_in_buf += docs;
        while self.buf.len() >= self.shard_tokens {
            let rest = self.buf.split_off(self.shard_tokens);
            self.flush_buf()?;
            self.buf = rest;
        }
        Ok(())
    }

    fn flush_buf(&mut self) -> anyhow::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let name = format!("{}_{:05}.bin", self.prefix, self.shards.len());
        let tmp = self.dir.join(format!("{name}.tmp"));
        let mut w = BufWriter::new(std::fs::File::create(&tmp)?);
        w.write_all(MAGIC)?;
        w.write_all(&(if self.wide { 4u32 } else { 2u32 }).to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
        w.write_all(&(self.buf.len() as u64).to_le_bytes())?;
        w.write_all(&self.docs_in_buf.to_le_bytes())?;
        for &t in &self.buf {
            if self.wide {
                w.write_all(&t.to_le_bytes())?;
            } else {
                w.write_all(&(t as u16).to_le_bytes())?;
            }
        }
        w.flush()?;
        drop(w);
        std::fs::rename(&tmp, self.dir.join(&name))?; // atomic: no half-written shards
        self.total_tokens += self.buf.len() as u64;
        self.shards.push((name, self.buf.len() as u64));
        self.buf.clear();
        self.docs_in_buf = 0;
        Ok(())
    }

    pub fn finish(mut self) -> anyhow::Result<(Vec<(String, u64)>, u64)> {
        self.flush_buf()?;
        Ok((self.shards, self.total_tokens))
    }
}

/// Read a `.bin` shard back (used by tests and tools).
pub fn read_shard(path: &Path) -> anyhow::Result<Vec<u32>> {
    let b = std::fs::read(path)?;
    anyhow::ensure!(b.len() >= HEADER_BYTES && &b[..8] == MAGIC, "not a NCVTOK01 shard");
    let width = u32::from_le_bytes(b[8..12].try_into()?) as usize;
    let n = u64::from_le_bytes(b[16..24].try_into()?) as usize;
    let body = &b[HEADER_BYTES..];
    anyhow::ensure!(body.len() == n * width, "truncated shard");
    Ok(match width {
        2 => body.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect(),
        4 => body.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(),
        w => anyhow::bail!("bad token width {w}"),
    })
}

#[derive(Debug)]
pub struct PrepareStats {
    pub split: String,
    pub files: usize,
    pub docs: u64,
    pub tokens: u64,
    pub shards: Vec<(String, u64)>,
}

/// Tokenize one split. `max_files` limits the number of input files (for quick runs).
pub fn prepare_split(
    tok: &Tokenizer,
    files: &[PathBuf],
    out: &Path,
    split: &str,
    shard_tokens: usize,
    mut progress: impl FnMut(&Path, u64, u64),
) -> anyhow::Result<PrepareStats> {
    std::fs::create_dir_all(out)?;
    let mut w = ShardWriter::new(out, split, shard_tokens, tok.vocab_size());
    let mut docs = 0u64;
    for f in files {
        for rg in read_text_row_groups(f)? {
            let toks = tokenize_docs(tok, &rg);
            docs += rg.len() as u64;
            w.push(&toks, rg.len() as u64)?;
        }
        progress(f, docs, w.total_tokens + w.buf.len() as u64);
    }
    let (shards, tokens) = w.finish()?;
    Ok(PrepareStats { split: split.into(), files: files.len(), docs, tokens, shards })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_writer_roundtrip_and_split() {
        let dir = std::env::temp_dir().join(format!("vlm_dataprep_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for vocab in [1000usize, 70000] {
            let mut w = ShardWriter::new(&dir, &format!("t{vocab}"), 7, vocab);
            let toks: Vec<u32> = (0..25).map(|i| (i * 2654) as u32 % vocab as u32).collect();
            w.push(&toks[..10], 1).unwrap();
            w.push(&toks[10..], 2).unwrap();
            let (shards, total) = w.finish().unwrap();
            assert_eq!(total, 25);
            assert_eq!(shards.iter().map(|s| s.1).collect::<Vec<_>>(), vec![7, 7, 7, 4]);
            let back: Vec<u32> = shards.iter().flat_map(|(n, _)| read_shard(&dir.join(n)).unwrap()).collect();
            assert_eq!(back, toks, "vocab {vocab}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
