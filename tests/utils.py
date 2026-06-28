import json
from pathlib import Path

def dump_regression_case(description: str, input_text: str, expected_typst: str, fixture_path: str = "tests/fixtures/test_db.json"):
    """
    Utility function to easily dump a failing case into the regression test database.
    Users can call this when they find a new edge case that breaks the PDF build.
    """
    path = Path(fixture_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cases = []
    if path.exists():
        with open(path, "r") as f:
            try:
                cases = json.load(f)
            except json.JSONDecodeError:
                pass

    cases.append({
        "description": description,
        "input": input_text,
        "expected_typst": expected_typst
    })

    with open(path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Added new regression case to {fixture_path}")
