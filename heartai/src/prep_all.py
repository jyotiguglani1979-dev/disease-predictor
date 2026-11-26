"""
src/prep_all.py
---------------
Runs all three data-prep steps. A missing dataset does not stop the others.

Run:  python -m src.prep_all
"""
from src import prep_brfss, prep_framingham, prep_uci
from src.db import tables


def main() -> None:
    results = {
        "UCI (Module A)": prep_uci.main(),
        "BRFSS (Module B)": prep_brfss.main(),
        "Framingham (Module C)": prep_framingham.main(),
    }
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {'OK     ' if ok else 'MISSING'}  {name}")
    print("\n  tables in database:", tables())


if __name__ == "__main__":
    main()
