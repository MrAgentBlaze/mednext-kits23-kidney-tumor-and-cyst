from pathlib import Path


def load_case_list(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Case split file not found: {path}"
        )

    cases = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return cases


if __name__ == "__main__":

    train_cases = load_case_list(
        "configs/train_cases.txt"
    )

    val_cases = load_case_list(
        "configs/val_cases.txt"
    )

    print(f"Training cases:   {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")

    overlap = set(train_cases) & set(val_cases)

    print(f"Case overlap:     {len(overlap)}")

    print()
    print("First 5 training cases:")
    for case in train_cases[:5]:
        print(f"  {case}")

    print()
    print("First 5 validation cases:")
    for case in val_cases[:5]:
        print(f"  {case}")