// Reject malformed saved annotations before loading data or appending labels.
use std::{env, fs, process::Command};

fn main() {
    let python = env::args().nth(1).unwrap_or(".venv/bin/python".into());
    let fixture = env::current_dir()
        .unwrap()
        .join(".tmp")
        .join(format!("annotation_check_{}", std::process::id()));
    fs::create_dir_all(&fixture).unwrap();
    let path = fixture.join("annotations.csv");
    let header = "item_id,headline,outlet,date,source_url,label,label_name,annotator\n";
    let valid = format!("{header}synthetic0,title,outlet,date,url,0,govt_critique,reviewer\nsynthetic1,title,outlet,date,url,,,reviewer\n");
    fs::write(&path, &valid).unwrap();
    let output = Command::new(&python).args([
        "-c",
        "from pathlib import Path; import sys; from scripts.annotate_local import load_done; print(','.join(sorted(load_done(Path(sys.argv[1])))))",
    ]).arg(&path).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "synthetic0");
    assert_eq!(fs::read_to_string(&path).unwrap(), valid);

    let malformed = [
        valid.replace("item_id,headline", "wrong_id,headline"),
        valid.replace(",0,govt_critique,", ",99,invalid,"),
        valid.replace(",0,govt_critique,", ",0,neutral,"),
        format!("{header}synthetic0,title\n"),
        format!("{valid}synthetic0,title,outlet,date,url,1,neutral,reviewer\n"),
    ];
    for contents in malformed {
        fs::write(&path, &contents).unwrap();
        let output = Command::new(&python)
            .args(["scripts/annotate_local.py", "--out"])
            .arg(&path)
            .env("HF_HUB_OFFLINE", "1")
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("cannot resume"));
        assert!(!String::from_utf8_lossy(&output.stdout).contains("Loading"));
        assert_eq!(fs::read_to_string(&path).unwrap(), contents);
    }
    fs::remove_dir_all(&fixture).unwrap();
    println!("Valid annotation resume passed; malformed files were rejected unchanged before data loading.");
}
