// Exercise the report CLI on committed summaries and a deliberately corrupted copy.
use std::{env, fs, path::Path, process::Command};

fn main() {
    let python = env::args()
        .nth(1)
        .unwrap_or_else(|| ".venv/bin/python".into());
    let reports = Path::new("reports/submission");
    let names = ["matched.json", "seeds.json", "early_stopping.json"];
    let originals: Vec<_> = names
        .iter()
        .map(|name| fs::read(reports.join(name)).unwrap())
        .collect();
    let output = Command::new(&python)
        .arg("scripts/verify_reports.py")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("\"matched_scores_recalculated\": 20"));
    for (name, original) in names.iter().zip(&originals) {
        assert_eq!(fs::read(reports.join(name)).unwrap(), *original);
    }

    let temporary = Path::new(".tmp").join(format!("report_check_{}", std::process::id()));
    let submission = temporary.join("submission");
    let verification = temporary.join("verification");
    fs::create_dir_all(&submission).unwrap();
    fs::create_dir_all(&verification).unwrap();
    for (name, original) in names.iter().zip(&originals) {
        fs::write(submission.join(name), original).unwrap();
    }
    for name in ["original_baselines.json", "input_recovery.json"] {
        let source = Path::new("reports/verification").join(name);
        if source.exists() {
            fs::copy(source, verification.join(name)).unwrap();
        }
    }
    let mut corrupted = String::from_utf8(originals[0].clone()).unwrap();
    let start = corrupted.find("\"macro_f1\": ").unwrap() + "\"macro_f1\": ".len();
    let end = start + corrupted[start..].find(',').unwrap();
    corrupted.replace_range(start..end, "0.999999");
    fs::write(submission.join(names[0]), corrupted).unwrap();
    let output = Command::new(&python)
        .args(["scripts/verify_reports.py", "--reports"])
        .arg(&submission)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("recalculated"));
    let recovery = verification.join("input_recovery.json");
    if recovery.exists() {
        fs::write(submission.join(names[0]), &originals[0]).unwrap();
        let original = fs::read_to_string(&recovery).unwrap();
        let changed = original.replace(
            "\"verified_selected_images\": 149",
            "\"verified_selected_images\": 148",
        );
        assert_ne!(original, changed);
        fs::write(recovery, changed).unwrap();
        let output = Command::new(&python)
            .args(["scripts/verify_reports.py", "--reports"])
            .arg(&submission)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("input recovery counts"));
    }
    fs::remove_dir_all(&temporary).unwrap();
    println!(
        "Report arithmetic passed, inputs were preserved, and a corrupted score was rejected."
    );
}
