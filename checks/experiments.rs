// Exercise resume, artifact validation, and aggregation on clearly synthetic records.
use std::{
    env, fs,
    path::Path,
    process::{Command, Output},
};

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.file_name() == "__pycache__" {
            continue;
        }
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn run(python: &Path, fixture: &Path, arguments: &[&str]) -> Output {
    Command::new(python)
        .args(arguments)
        .current_dir(fixture)
        .env("PYTHONPATH", fixture.join("src"))
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap()
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run_directories(fixture: &Path) -> usize {
    fs::read_dir(fixture.join("runs/all_headlines"))
        .unwrap()
        .count()
}

fn main() {
    let root = env::current_dir().unwrap();
    let python = root.join(env::args().nth(1).unwrap_or(".venv/bin/python".into()));
    let fixture = root
        .join(".tmp")
        .join(format!("experiment_check_{}", std::process::id()));
    fs::create_dir(&fixture).unwrap();
    copy_tree(&root.join("src"), &fixture.join("src"));
    copy_tree(&root.join("configs"), &fixture.join("configs"));
    fs::create_dir(fixture.join("scripts")).unwrap();
    for name in [
        "run_matched_sweep.py",
        "matched_analysis.py",
        "seed_analysis.py",
        "input_manifest.py",
    ] {
        fs::copy(
            root.join("scripts").join(name),
            fixture.join("scripts").join(name),
        )
        .unwrap();
    }
    fs::create_dir_all(fixture.join("data/processed")).unwrap();
    let mut csv = "item_id,title,text,label,label_name,source_url,source_index,image_path,has_image,is_augmented\n".to_owned();
    let labels = ["govt_critique", "neutral", "govt_leaning"];
    for i in 0..15 {
        csv.push_str(&format!("synthetic_{i},Synthetic headline {i},Synthetic headline {i},{},{},https://example.invalid/{i},{i},,False,False\n", i % 3, labels[i % 3]));
    }
    fs::write(fixture.join("data/processed/corpus.csv"), csv).unwrap();
    success(&run(
        &python,
        &fixture,
        &["scripts/input_manifest.py", "--write"],
    ));
    success(&run(&python, &fixture, &["scripts/input_manifest.py"]));
    let corpus_path = fixture.join("data/processed/corpus.csv");
    let original_csv = fs::read(&corpus_path).unwrap();
    let mut changed_csv = original_csv.clone();
    changed_csv.extend_from_slice(b"\n");
    fs::write(&corpus_path, changed_csv).unwrap();
    assert!(!run(&python, &fixture, &["scripts/input_manifest.py"])
        .status
        .success());
    fs::write(&corpus_path, original_csv).unwrap();
    let args = [
        "scripts/run_matched_sweep.py",
        "--only",
        "majority",
        "--population",
        "all_headlines",
        "--seed",
        "42",
        "--out",
        "runs",
    ];
    success(&run(&python, &fixture, &args));
    assert_eq!(run_directories(&fixture), 1);
    let repeated = run(&python, &fixture, &args);
    success(&repeated);
    assert!(String::from_utf8_lossy(&repeated.stdout).contains("matching run"));
    assert_eq!(run_directories(&fixture), 1);

    success(&run(
        &python,
        &fixture,
        &["-c", "import json; from pathlib import Path; p=Path('runs/sweep_summary.json'); rows=json.loads(p.read_text()); rows[0]['macro_f1']=-1; p.write_text(json.dumps(rows))"],
    ));
    success(&run(&python, &fixture, &args));
    assert_eq!(run_directories(&fixture), 1);
    success(&run(
        &python,
        &fixture,
        &["scripts/seed_analysis.py", "runs", "--dest", "repaired_summary"],
    ));

    let mut changed = args;
    changed[6] = "43";
    success(&run(&python, &fixture, &changed));
    assert_eq!(run_directories(&fixture), 2);
    let summary = fs::read_to_string(fixture.join("runs/sweep_summary.json")).unwrap();
    let token = "\"run_dir\": \"";
    let start = summary.find(token).unwrap() + token.len();
    let end = start + summary[start..].find('"').unwrap();
    let prediction = fixture
        .join("runs")
        .join(&summary[start..end])
        .join("predictions.csv");
    let original = fs::read(&prediction).unwrap();
    let mut corrupted = original.clone();
    corrupted.extend_from_slice(b"deliberate synthetic corruption\n");
    fs::write(&prediction, &corrupted).unwrap();
    let rejected = run(
        &python,
        &fixture,
        &["scripts/seed_analysis.py", "runs", "--dest", "aggregate"],
    );
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("checksum mismatch"));
    success(&run(&python, &fixture, &changed));
    assert_eq!(run_directories(&fixture), 3);
    assert_eq!(fs::read(&prediction).unwrap(), corrupted);

    success(&run(
        &python,
        &fixture,
        &["scripts/seed_analysis.py", "runs", "--dest", "aggregate"],
    ));
    let aggregate = fs::read_to_string(fixture.join("aggregate/seeds.json")).unwrap();
    assert!(!aggregate.contains("NaN"));
    assert!(aggregate.contains("\"identity\""));
    let duplicate = run(
        &python,
        &fixture,
        &[
            "scripts/seed_analysis.py",
            "runs",
            "runs",
            "--dest",
            "duplicate",
        ],
    );
    assert!(!duplicate.status.success());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("duplicate seed"));

    for name in ["mbert", "xlmr"] {
        fs::write(fixture.join("configs/text").join(format!("{name}.yaml")),
            format!("name: {name}\nfamily: hf_text_classifier\nmodality: text\npretrained: synthetic/missing_{name}\n")).unwrap();
    }
    success(&run(
        &python,
        &fixture,
        &[
            "scripts/run_matched_sweep.py",
            "--only",
            "majority",
            "mbert",
            "xlmr",
            "--population",
            "all_headlines",
            "--out",
            "failed_runs",
        ],
    ));
    let failures = fs::read_to_string(fixture.join("failed_runs/sweep_summary.json")).unwrap();
    assert_eq!(failures.matches("\"status\": \"failed\"").count(), 2);
    assert!(failures.contains("\"name\": \"xlmr\""));
    fs::remove_dir_all(&fixture).unwrap();
    println!("Synthetic experiment resume, changed seed, checksum rejection, preservation, and aggregation passed.");
}
