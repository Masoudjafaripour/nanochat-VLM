//! CORE metric (DCLM), ported from `nanochat_vlm/core_eval.py` and `src/base_eval.py`.

use crate::forward_losses;
use crate::pyrandom::PyRandom;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;
use vlm_inference::model::Gpt;
use vlm_inference::tokenizer::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TaskType {
    MultipleChoice,
    Schema,
    LanguageModeling,
}

#[derive(Clone, Debug)]
pub struct TaskMeta {
    pub label: String,
    pub task_type: TaskType,
    pub dataset_uri: String,
    pub num_fewshot: usize,
    pub delim: String,
}

fn s(v: &Value) -> String {
    match v {
        Value::String(x) => x.clone(),
        other => other.to_string(), // Jinja would str() non-strings
    }
}

fn gold(v: &Value) -> usize {
    v["gold"].as_u64().unwrap_or(0) as usize
}

/// Python str.strip() (Jinja's `trim` filter does the same).
fn strip(x: &str) -> &str {
    x.trim_matches(|c: char| c.is_whitespace())
}

/// The three Jinja templates, rendered by hand. Each few-shot example is followed by "\n\n".
pub fn render(task: &TaskMeta, item: &Value, shots: &[&Value]) -> Vec<String> {
    let d = &task.delim;
    match task.task_type {
        TaskType::MultipleChoice => {
            let pre: String = shots.iter().map(|e| format!("{}{d}{}\n\n", s(&e["query"]), s(&e["choices"][gold(e)]))).collect();
            item["choices"].as_array().unwrap().iter().map(|c| format!("{pre}{}{d}{}", s(&item["query"]), s(c))).collect()
        }
        TaskType::Schema => {
            let pre: String =
                shots.iter().map(|e| format!("{}{d}{}\n\n", s(&e["context_options"][gold(e)]), s(&e["continuation"]))).collect();
            item["context_options"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| format!("{pre}{}{d}{}", s(c), s(&item["continuation"])))
                .collect()
        }
        TaskType::LanguageModeling => {
            let pre: String =
                shots.iter().map(|e| format!("{}{d}{}\n\n", strip(&s(&e["context"])), s(&e["continuation"]))).collect();
            let ctx = strip(&s(&item["context"])).to_string();
            let without = format!("{pre}{ctx}{d}");
            let with = format!("{pre}{ctx}{d}{}", s(&item["continuation"]));
            // core_eval.py strips prompt_without only
            vec![strip(&without).to_string(), with]
        }
    }
}

fn common_len(seqs: &[Vec<u32>], from_left: bool) -> usize {
    let min = seqs.iter().map(|x| x.len()).min().unwrap_or(0);
    for i in 0..min {
        let at = |x: &Vec<u32>| if from_left { x[i] } else { x[x.len() - 1 - i] };
        let t0 = at(&seqs[0]);
        if !seqs.iter().all(|x| at(x) == t0) {
            return i;
        }
    }
    min
}

/// `evaluate_example`: true if the model gets item `idx` right.
/// Few-shot selection (random.Random(1234 + idx).sample) + rendering for item `idx`.
pub fn prompts_for(data: &[Value], idx: usize, task: &TaskMeta) -> Vec<String> {
    let shots: Vec<&Value> = if task.num_fewshot > 0 {
        let mut rng = PyRandom::new(1234 + idx as u64);
        let avail: Vec<usize> = (0..data.len()).filter(|&i| i != idx).collect();
        rng.sample_indices(avail.len(), task.num_fewshot).into_iter().map(|j| &data[avail[j]]).collect()
    } else {
        vec![]
    };
    render(task, &data[idx], &shots)
}

pub fn evaluate_example(model: &Gpt, tok: &Tokenizer, data: &[Value], idx: usize, task: &TaskMeta) -> anyhow::Result<bool> {
    let item = &data[idx];
    let prompts = prompts_for(data, idx, task);
    let bos = tok.bos_id();
    let mut tokens: Vec<Vec<u32>> = prompts.iter().map(|p| std::iter::once(bos).chain(tok.encode(p)).collect()).collect();
    let (starts, ends): (Vec<usize>, Vec<usize>) = match task.task_type {
        TaskType::MultipleChoice => {
            let a = common_len(&tokens, true);
            tokens.iter().map(|x| (a, x.len())).unzip()
        }
        TaskType::Schema => {
            let sfx = common_len(&tokens, false);
            tokens.iter().map(|x| (x.len() - sfx, x.len())).unzip()
        }
        TaskType::LanguageModeling => {
            let (wo, w) = (tokens[0].clone(), tokens[1].clone());
            anyhow::ensure!(wo.len() < w.len() && w[..wo.len()] == wo[..], "prompt without is not a prefix of prompt with");
            tokens = vec![w.clone()];
            (vec![wo.len()], vec![w.len()])
        }
    };
    // stack_sequences: right-pad with BOS
    let tmax = tokens.iter().map(|x| x.len()).max().unwrap();
    let padded: Vec<Vec<u32>> = tokens.iter().map(|x| {
        let mut v = x.clone();
        v.resize(tmax, bos);
        v
    }).collect();
    let (losses, preds) = forward_losses(model, &padded)?;
    Ok(match task.task_type {
        TaskType::LanguageModeling => {
            let (si, ei) = (starts[0], ends[0]);
            (si..ei).all(|i| preds[0][i - 1] == padded[0][i])
        }
        _ => {
            // lowest mean loss over the continuation; first index wins ties (Python list.index(min))
            let means: Vec<f32> = starts.iter().zip(&ends).enumerate().map(|(r, (&si, &ei))| {
                let sl = &losses[r][si - 1..ei - 1];
                sl.iter().sum::<f32>() / sl.len() as f32
            }).collect();
            let mut best = 0;
            for (i, &m) in means.iter().enumerate() {
                if m < means[best] {
                    best = i;
                }
            }
            best == gold(item)
        }
    })
}

pub fn evaluate_task(model: &Gpt, tok: &Tokenizer, data: &[Value], task: &TaskMeta) -> anyhow::Result<f64> {
    let mut correct = 0usize;
    for idx in 0..data.len() {
        correct += evaluate_example(model, tok, data, idx, task)? as usize;
    }
    Ok(correct as f64 / data.len() as f64)
}

#[derive(Deserialize)]
struct CoreYaml {
    icl_tasks: Vec<YamlTask>,
}

#[derive(Deserialize)]
struct YamlTask {
    label: String,
    dataset_uri: String,
    num_fewshot: Vec<usize>,
    icl_task_type: String,
    #[serde(default)]
    continuation_delimiter: Option<String>,
}

/// Load `core.yaml` + `eval_meta_data.csv` (Random baseline per task) from an eval bundle.
pub fn load_bundle(dir: &Path) -> anyhow::Result<Vec<(TaskMeta, f64)>> {
    let y: CoreYaml = serde_yaml::from_str(&std::fs::read_to_string(dir.join("core.yaml"))?)?;
    let csv = std::fs::read_to_string(dir.join("eval_meta_data.csv"))?;
    let rows: Vec<Vec<String>> = csv.lines().filter(|l| !l.trim().is_empty()).map(split_csv).collect();
    let col = |name: &str| rows[0].iter().position(|h| h.trim() == name).ok_or_else(|| anyhow::anyhow!("csv lacks {name}"));
    let (ti, bi) = (col("Eval Task")?, col("Random baseline")?);
    y.icl_tasks
        .into_iter()
        .map(|t| {
            let task_type = match t.icl_task_type.as_str() {
                "multiple_choice" => TaskType::MultipleChoice,
                "schema" => TaskType::Schema,
                "language_modeling" => TaskType::LanguageModeling,
                o => anyhow::bail!("unsupported task type {o}"),
            };
            let baseline = rows[1..]
                .iter()
                .find(|r| r[ti] == t.label)
                .ok_or_else(|| anyhow::anyhow!("no baseline for {}", t.label))?[bi]
                .trim()
                .parse::<f64>()?;
            let meta = TaskMeta {
                label: t.label,
                task_type,
                dataset_uri: t.dataset_uri,
                num_fewshot: t.num_fewshot[0],
                delim: t.continuation_delimiter.unwrap_or_else(|| " ".into()),
            };
            Ok((meta, baseline))
        })
        .collect()
}

fn split_csv(line: &str) -> Vec<String> {
    let (mut out, mut cur, mut q) = (vec![], String::new(), false);
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if q && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => q = !q,
            ',' if !q => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

#[derive(Debug, serde::Serialize)]
pub struct CoreResult {
    pub results: Vec<(String, f64)>,
    pub centered: Vec<(String, f64)>,
    pub core_metric: f64,
}

/// `base_eval.evaluate_model`: shuffle each task with random.Random(1337), optional cap,
/// accuracy, centered by the random baseline, CORE = mean of centered scores.
pub fn evaluate_bundle(
    model: &Gpt,
    tok: &Tokenizer,
    dir: &Path,
    max_per_task: Option<usize>,
    mut dump: Option<&mut dyn std::io::Write>,
    mut log: impl FnMut(&str, f64, f64),
) -> anyhow::Result<CoreResult> {
    let mut r = CoreResult { results: vec![], centered: vec![], core_metric: 0.0 };
    for (task, baseline) in load_bundle(dir)? {
        let path = dir.join("eval_data").join(&task.dataset_uri);
        let mut data: Vec<Value> = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        PyRandom::new(1337).shuffle(&mut data);
        if let Some(m) = max_per_task {
            data.truncate(m);
        }
        if let Some(w) = dump.as_deref_mut() {
            for idx in 0..data.len() {
                writeln!(w, "{}", serde_json::json!({"label": task.label, "idx": idx, "prompts": prompts_for(&data, idx, &task)}))?;
            }
        }
        let acc = evaluate_task(model, tok, &data, &task)?;
        let c = (acc - 0.01 * baseline) / (1.0 - 0.01 * baseline);
        log(&task.label, acc, c);
        r.results.push((task.label.clone(), acc));
        r.centered.push((task.label, c));
    }
    r.core_metric = r.centered.iter().map(|x| x.1).sum::<f64>() / r.centered.len().max(1) as f64;
    Ok(r)
}
