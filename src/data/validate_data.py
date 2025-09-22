# src/data/validate_data.py
# Great Expectations validation for Kvasir-SEG using the current Fluent API (GE 1.x)

import os
from pathlib import Path

import pandas as pd

import great_expectations as gx

# --- Setup paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Kvasir-SEG"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"


def _assert_data_dirs():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")
    if not MASKS_DIR.exists():
        raise FileNotFoundError(f"Masks directory not found: {MASKS_DIR}")


def _build_index_df() -> pd.DataFrame:
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))]
    mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith((".jpg", ".png"))]

    mask_ids = {os.path.splitext(m)[0] for m in mask_files}

    df = pd.DataFrame(
        {
            "image_id": [os.path.splitext(f)[0] for f in image_files],
            "has_mask": [os.path.splitext(f)[0] in mask_ids for f in image_files],
        }
    )
    return df


def main():
    # 0) Prepare data frame
    _assert_data_dirs()
    df = _build_index_df()

    # 1) Context (file-based project under PROJECT_ROOT/great_expectations)
    # Create a Fluent context configured to persist locally under great_expectations/
    from great_expectations.data_context.types.base import (
        DataContextConfig,
        FilesystemStoreBackendDefaults,
    )

    project_config = DataContextConfig(
        store_backend_defaults=FilesystemStoreBackendDefaults(
            root_directory=str(PROJECT_ROOT / "gx")
        )
    )
    context = gx.get_context(project_config=project_config)
    print("Using Great Expectations context at:", PROJECT_ROOT / "gx")

    # 2) Datasource and in-memory DataFrame asset (Fluent API)
    datasource_name = "kvasir_pandas_source"
    data_asset_name = "images_index"

    data_source = context.data_sources.add_or_update_pandas(name=datasource_name)
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)
    batch_request = data_asset.build_batch_request(options={"dataframe": df})

    # 3) Expectation Suite (FileDataContext suites manager in GE 1.6.x)
    suite_name = "raw_kvasir_suite"
    try:
        context.suites.get(name=suite_name)
    except Exception:
        context.suites.add(gx.ExpectationSuite(name=suite_name))

    # 4) Validator + expectations
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    print("Adding expectations...")
    # Kvasir-SEG has 1,000 images with masks
    validator.expect_table_row_count_to_equal(1000)
    validator.expect_column_values_to_be_in_set(column="has_mask", value_set=[True])
    validator.expect_column_values_to_be_unique(column="image_id")

    # Persist expectations to the store
    context.suites.add_or_update(validator.get_expectation_suite())

    # 5) Run validation directly (Checkpoints API not available on FileDataContext in 1.6.x)
    print("Running validation...")
    result = validator.validate()

    if not result.success:
        print("Validation failed!")
        raise RuntimeError(
            "Data validation failed. Check the logs or open Data Docs for details."
        )

    print("Validation successful!")
    context.build_data_docs()
    docs_index = (
        PROJECT_ROOT / "gx" / "uncommitted" / "data_docs" / "local_site" / "index.html"
    )
    print("\nTo view the detailed validation report, open this file in your browser:")
    print(f"file://{docs_index}")


if __name__ == "__main__":
    main()
