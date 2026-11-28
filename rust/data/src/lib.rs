//! Streaming multimodal data pipeline.
//!
//! Input: a JSONL manifest, one sample per line:
//! `{"image": "relative/or/absolute/path.png", "text": "a caption"}`
//! (relative paths resolve against the manifest's directory).
//!
//! The loader:
//! 1. shards the index space by `(rank, world_size)` (same convention as the DDP training scripts),
//! 2. shuffles each epoch with a seeded RNG,
//! 3. decodes, resizes and patchifies images and tokenizes text on `num_workers` threads,
//! 4. restores the deterministic sample order with a small reorder buffer,
//! 5. hands out fixed-shape batches through a bounded channel (`prefetch` batches ahead),
//!    so memory stays bounded however large the dataset is.
//!
//! Tokenization is injected as a closure, so this crate doesn't depend on the model crates.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use vlm_preprocessing::{preprocess_path, PreprocessConfig};

#[derive(Clone, Debug, Deserialize)]
pub struct Record {
    pub image: String,
    pub text: String,
}

/// Parsed manifest with resolved image paths.
#[derive(Clone, Debug)]
pub struct Manifest {
    pub records: Vec<(PathBuf, String)>,
}

impl Manifest {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let f = std::fs::File::open(path)?;
        let mut records = Vec::new();
        for (i, line) in BufReader::new(f).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let r: Record = serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("{}:{}: {e}", path.display(), i + 1))?;
            let p = PathBuf::from(&r.image);
            records.push((if p.is_absolute() { p } else { base.join(p) }, r.text));
        }
        Ok(Self { records })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

pub type TextEncoder = Arc<dyn Fn(&str) -> Vec<u32> + Send + Sync>;

#[derive(Clone, Debug)]
pub struct LoaderConfig {
    pub batch_size: usize,
    pub num_workers: usize,
    /// Batches buffered ahead of the consumer.
    pub prefetch: usize,
    pub shuffle: bool,
    pub seed: u64,
    pub rank: usize,
    pub world_size: usize,
    /// Text is truncated to this many tokens; batches are padded to the longest sample.
    pub max_text_tokens: usize,
    pub pad_id: u32,
    pub drop_last: bool,
    /// None = loop forever (training); Some(n) = stop after n epochs.
    pub epochs: Option<usize>,
    /// Skip unreadable/corrupt samples instead of failing the stream.
    pub skip_errors: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            num_workers: 4,
            prefetch: 4,
            shuffle: true,
            seed: 0,
            rank: 0,
            world_size: 1,
            max_text_tokens: 128,
            pad_id: 0,
            drop_last: true,
            epochs: Some(1),
            skip_errors: true,
        }
    }
}

/// A collated batch.
#[derive(Debug)]
pub struct Batch {
    pub size: usize,
    pub num_patches: usize,
    pub patch_dim: usize,
    /// (size, num_patches, patch_dim), row-major.
    pub patches: Vec<f32>,
    /// (size, max_len) padded with `pad_id`.
    pub input_ids: Vec<u32>,
    pub max_len: usize,
    pub lengths: Vec<usize>,
    /// Manifest indices of the samples (for debugging / exact resumption).
    pub indices: Vec<usize>,
    pub epoch: usize,
}

struct Sample {
    index: usize,
    patches: Vec<f32>,
    ids: Vec<u32>,
}

#[derive(Debug, Default)]
pub struct LoaderStats {
    pub samples_ok: AtomicUsize,
    pub samples_skipped: AtomicUsize,
}

pub struct Loader {
    rx: Receiver<anyhow::Result<Batch>>,
    stop: Arc<AtomicBool>,
    handles: Vec<JoinHandle<()>>,
    pub stats: Arc<LoaderStats>,
    pub batches_per_epoch: usize,
}

/// Indices owned by this rank for one epoch (interleaved sharding, like DistributedSampler).
pub fn epoch_indices(n: usize, cfg: &LoaderConfig, epoch: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    if cfg.shuffle {
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.wrapping_add(epoch as u64));
        idx.shuffle(&mut rng);
    }
    idx.into_iter().skip(cfg.rank).step_by(cfg.world_size.max(1)).collect()
}

impl Loader {
    pub fn new(
        manifest: Arc<Manifest>,
        pre: PreprocessConfig,
        encode: TextEncoder,
        cfg: LoaderConfig,
    ) -> anyhow::Result<Self> {
        pre.validate()?;
        anyhow::ensure!(cfg.batch_size > 0 && cfg.num_workers > 0, "batch_size and num_workers must be > 0");
        anyhow::ensure!(cfg.rank < cfg.world_size.max(1), "rank must be < world_size");
        let per_rank = epoch_indices(manifest.len(), &cfg, 0).len();
        let batches_per_epoch =
            if cfg.drop_last { per_rank / cfg.batch_size } else { per_rank.div_ceil(cfg.batch_size) };

        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(LoaderStats::default());
        // Work queue: (sequence number, epoch, manifest index). Workers pull from it.
        let (work_tx, work_rx) = sync_channel::<(usize, usize, usize)>(cfg.batch_size * cfg.prefetch * 2);
        let work_rx = Arc::new(Mutex::new(work_rx));
        let (sample_tx, sample_rx) = sync_channel::<(usize, usize, Option<Sample>, Option<String>)>(
            cfg.batch_size * cfg.prefetch * 2,
        );
        let (batch_tx, batch_rx) = sync_channel::<anyhow::Result<Batch>>(cfg.prefetch.max(1));
        let mut handles = Vec::new();

        // Producer: emits the index stream epoch by epoch.
        {
            let (stop, cfg, n) = (stop.clone(), cfg.clone(), manifest.len());
            handles.push(std::thread::spawn(move || {
                let mut seq = 0;
                let mut epoch = 0;
                while cfg.epochs.is_none_or(|e| epoch < e) {
                    let idx = epoch_indices(n, &cfg, epoch);
                    if idx.is_empty() {
                        break;
                    }
                    for i in idx {
                        if stop.load(Ordering::Relaxed) || work_tx.send((seq, epoch, i)).is_err() {
                            return;
                        }
                        seq += 1;
                    }
                    epoch += 1;
                }
            }));
        }

        // Workers: preprocess image + tokenize text.
        for _ in 0..cfg.num_workers {
            let (work_rx, sample_tx, manifest, pre, encode, stop, max_tok) = (
                work_rx.clone(),
                sample_tx.clone(),
                manifest.clone(),
                pre.clone(),
                encode.clone(),
                stop.clone(),
                cfg.max_text_tokens,
            );
            handles.push(std::thread::spawn(move || loop {
                let job = work_rx.lock().unwrap().recv();
                let Ok((seq, epoch, i)) = job else { return };
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                let (path, text) = &manifest.records[i];
                let msg = match preprocess_path(path, &pre) {
                    Ok(p) => {
                        let mut ids = encode(text);
                        ids.truncate(max_tok);
                        (seq, epoch, Some(Sample { index: i, patches: p.data, ids }), None)
                    }
                    Err(e) => (seq, epoch, None, Some(format!("{}: {e}", path.display()))),
                };
                if sample_tx.send(msg).is_err() {
                    return;
                }
            }));
        }
        drop(sample_tx);

        // Collator: reorder by sequence number, group into batches.
        {
            let (cfg, stats, stop) = (cfg.clone(), stats.clone(), stop.clone());
            let (np, pd) = (pre.num_patches(), pre.patch_dim());
            handles.push(std::thread::spawn(move || {
                let mut pending: BTreeMap<usize, (usize, Option<Sample>, Option<String>)> = BTreeMap::new();
                let mut next = 0usize;
                let mut cur: Vec<Sample> = Vec::with_capacity(cfg.batch_size);
                let mut cur_epoch = 0usize;
                let flush = |cur: &mut Vec<Sample>, epoch: usize| -> Option<Batch> {
                    if cur.is_empty() {
                        return None;
                    }
                    let b = collate(std::mem::take(cur), np, pd, cfg.pad_id, epoch);
                    Some(b)
                };
                for (seq, epoch, s, err) in sample_rx.iter() {
                    pending.insert(seq, (epoch, s, err));
                    while let Some((epoch, s, err)) = pending.remove(&next) {
                        next += 1;
                        if epoch != cur_epoch {
                            // epoch boundary: emit or drop the partial batch
                            if !cfg.drop_last {
                                if let Some(b) = flush(&mut cur, cur_epoch) {
                                    if batch_tx.send(Ok(b)).is_err() {
                                        return;
                                    }
                                }
                            }
                            cur.clear();
                            cur_epoch = epoch;
                        }
                        match (s, err) {
                            (Some(s), _) => {
                                stats.samples_ok.fetch_add(1, Ordering::Relaxed);
                                cur.push(s);
                            }
                            (None, Some(e)) => {
                                stats.samples_skipped.fetch_add(1, Ordering::Relaxed);
                                if !cfg.skip_errors {
                                    let _ = batch_tx.send(Err(anyhow::anyhow!(e)));
                                    stop.store(true, Ordering::Relaxed);
                                    return;
                                }
                            }
                            _ => unreachable!(),
                        }
                        if cur.len() == cfg.batch_size {
                            let b = flush(&mut cur, cur_epoch).unwrap();
                            if batch_tx.send(Ok(b)).is_err() {
                                return;
                            }
                        }
                    }
                }
                if !cfg.drop_last {
                    if let Some(b) = flush(&mut cur, cur_epoch) {
                        let _ = batch_tx.send(Ok(b));
                    }
                }
            }));
        }

        Ok(Self { rx: batch_rx, stop, handles, stats, batches_per_epoch })
    }
}

fn collate(samples: Vec<Sample>, np: usize, pd: usize, pad: u32, epoch: usize) -> Batch {
    let size = samples.len();
    let max_len = samples.iter().map(|s| s.ids.len()).max().unwrap_or(0);
    let mut patches = Vec::with_capacity(size * np * pd);
    let mut input_ids = vec![pad; size * max_len];
    let mut lengths = Vec::with_capacity(size);
    let mut indices = Vec::with_capacity(size);
    for (b, s) in samples.into_iter().enumerate() {
        patches.extend_from_slice(&s.patches);
        input_ids[b * max_len..b * max_len + s.ids.len()].copy_from_slice(&s.ids);
        lengths.push(s.ids.len());
        indices.push(s.index);
    }
    Batch { size, num_patches: np, patch_dim: pd, patches, input_ids, max_len, lengths, indices, epoch }
}

impl Iterator for Loader {
    type Item = anyhow::Result<Batch>;
    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Drain so blocked senders wake up and observe the closed channel.
        let rx = std::mem::replace(&mut self.rx, sync_channel(1).1);
        drop(rx);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_dataset(dir: &Path, n: usize, corrupt: &[usize]) -> PathBuf {
        let mut lines = String::new();
        for i in 0..n {
            let name = format!("img_{i}.png");
            if corrupt.contains(&i) {
                std::fs::write(dir.join(&name), b"not a png").unwrap();
            } else {
                let img = image::RgbImage::from_pixel(40 + i as u32, 30, image::Rgb([i as u8, 0, 255]));
                img.save(dir.join(&name)).unwrap();
            }
            lines.push_str(&format!("{{\"image\": \"{name}\", \"text\": \"caption {i}\"}}\n"));
        }
        let m = dir.join("manifest.jsonl");
        std::fs::write(&m, lines).unwrap();
        m
    }

    fn enc() -> TextEncoder {
        Arc::new(|s: &str| s.bytes().map(|b| b as u32).collect())
    }

    fn small() -> PreprocessConfig {
        PreprocessConfig { image_size: (32, 32), patch_size: 16, ..Default::default() }
    }

    #[test]
    fn deterministic_order_and_full_coverage() {
        let dir = std::env::temp_dir().join(format!("vlm_data_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let m = Arc::new(Manifest::open(write_dataset(&dir, 23, &[])).unwrap());
        let run = |workers| {
            let cfg = LoaderConfig { batch_size: 4, num_workers: workers, drop_last: false, ..Default::default() };
            Loader::new(m.clone(), small(), enc(), cfg).unwrap().map(|b| b.unwrap().indices).collect::<Vec<_>>()
        };
        let a = run(1);
        let b = run(7);
        assert_eq!(a, b, "order must not depend on worker count");
        let mut all: Vec<usize> = a.concat();
        all.sort();
        assert_eq!(all, (0..23).collect::<Vec<_>>());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sharding_skips_and_padding() {
        let dir = std::env::temp_dir().join(format!("vlm_data_test2_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let m = Arc::new(Manifest::open(write_dataset(&dir, 20, &[3])).unwrap());
        let mut seen = vec![];
        for rank in 0..2 {
            let cfg = LoaderConfig { batch_size: 3, rank, world_size: 2, drop_last: false, ..Default::default() };
            let loader = Loader::new(m.clone(), small(), enc(), cfg).unwrap();
            let stats = loader.stats.clone();
            for b in loader {
                let b = b.unwrap();
                assert_eq!(b.patches.len(), b.size * 4 * 768);
                for (k, &len) in b.lengths.iter().enumerate() {
                    let row = &b.input_ids[k * b.max_len..(k + 1) * b.max_len];
                    assert!(row[len..].iter().all(|&t| t == 0));
                }
                seen.extend(b.indices);
            }
            let _ = stats;
        }
        seen.sort();
        let expect: Vec<usize> = (0..20).filter(|&i| i != 3).collect();
        assert_eq!(seen, expect, "ranks are disjoint, cover everything, corrupt sample skipped");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn early_drop_does_not_hang() {
        let dir = std::env::temp_dir().join(format!("vlm_data_test3_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let m = Arc::new(Manifest::open(write_dataset(&dir, 10, &[])).unwrap());
        let cfg = LoaderConfig { batch_size: 2, epochs: None, prefetch: 1, ..Default::default() };
        let mut loader = Loader::new(m, small(), enc(), cfg).unwrap();
        for _ in 0..12 {
            loader.next().unwrap().unwrap(); // infinite stream crosses epochs
        }
        drop(loader);
        std::fs::remove_dir_all(&dir).ok();
    }
}
