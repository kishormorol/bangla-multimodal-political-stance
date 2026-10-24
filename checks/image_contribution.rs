use std::{
    env, fs,
    path::Path,
    process::{Command, Output},
};

fn run(python: &Path, root: &Path, fixture: &Path, script: &str) -> Output {
    Command::new(python)
        .args(["-c", script])
        .current_dir(root)
        .env("PYTHONPATH", root.join("src"))
        .env("HF_HUB_OFFLINE", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("BMPB_CHECK_OUT", fixture)
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

fn main() {
    let root = env::current_dir().unwrap();
    let python = root.join(env::args().nth(1).unwrap_or(".venv-gpu/bin/python".into()));
    let fixture = root
        .join(".tmp")
        .join(format!("image_contribution_check_{}", std::process::id()));
    fs::create_dir(&fixture).unwrap();
    let script = r#"
import os
from pathlib import Path

import numpy as np
import pandas as pd

from bmpb.config import load_yaml
from bmpb.crossval import make_folds
from scripts.image_contribution import fit_conditions

out = Path(os.environ["BMPB_CHECK_OUT"])
labels = np.tile([0, 1, 2], 5)
frame = pd.DataFrame({
    "item_id": [f"synthetic_{i}" for i in range(15)],
    "text": ["the same synthetic headline"] * 15,
    "label": labels,
    "group": [f"group_{i}" for i in range(15)],
    "source_index": [f"source_{i}" for i in range(15)],
    "source_url": [f"https://example.invalid/{i}" for i in range(15)],
})
features = np.eye(3)[labels]
folds = make_folds(frame, 5, "source_index", 42)
config = load_yaml(Path("configs/ablations/image_contribution.yaml"))
frames, scores = fit_conditions(frame, features, folds, config, 42, out)
expected = set(frame.item_id)
reference = None
for name in "ABC":
    result = frames[name]
    csv = pd.read_csv(out / f"seed_42_{name}.csv")
    for candidate in (result, csv):
        assert len(candidate) == 15 and candidate.item_id.is_unique
        assert set(candidate.item_id) == expected
    identity = result.set_index("item_id")[ ["fold", "group", "label"] ].sort_index()
    assert csv.set_index("item_id")[ ["fold", "group", "label"] ].sort_index().equals(identity)
    if reference is None:
        reference = identity
    else:
        assert identity.equals(reference)
assert scores["B"]["macro_f1"] == 1.0, scores["B"]
assert scores["C"]["macro_f1"] == 1.0, scores["C"]
assert scores["A"]["macro_f1"] < scores["B"]["macro_f1"], scores["A"]

before = {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
duplicate = frame.copy()
duplicate.loc[1, "item_id"] = duplicate.loc[0, "item_id"]
for invalid_frame, invalid_folds in ((duplicate, folds), (frame, [*folds[:-1], np.r_[folds[-1], folds[0][0]]])):
    try:
        fit_conditions(invalid_frame, features, invalid_folds, config, 42, out)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input was accepted")
    assert {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()} == before
print("Synthetic image contribution checks passed.")
"#;
    success(&run(&python, &root, &fixture, script));
    fs::remove_dir_all(&fixture).unwrap();
    println!("Synthetic image contribution success and rejection checks passed.");
}
