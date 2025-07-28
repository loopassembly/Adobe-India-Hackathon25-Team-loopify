#!/bin/bash

# Schema validation script for PDF outline extraction
# Validates all JSON outputs against the required schema

echo "🔍 Validating JSON outputs against schema..."

python3 - <<'PY'
import json
import sys
import glob
import jsonschema
from pathlib import Path

def validate_outputs():
    # Load schema
    schema_path = Path('sample_dataset/schema/output_schema.json')
    if not schema_path.exists():
        print("❌ Schema file not found at sample_dataset/schema/output_schema.json")
        sys.exit(1)

    with open(schema_path) as f:
        schema = json.load(f)

    # Find output files
    output_patterns = ['out/*.json', 'output/*.json', 'docker_out/*.json']
    json_files = []

    for pattern in output_patterns:
        json_files.extend(glob.glob(pattern))

    if not json_files:
        print("❌ No JSON files found in out/, output/, or docker_out/ directories")
        sys.exit(1)

    print(f"📁 Found {len(json_files)} JSON files to validate")

    # Validate each file
    errors = []
    for json_file in sorted(json_files):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Validate against schema
            jsonschema.validate(instance=data, schema=schema)
            print(f"✅ {json_file}: VALID")

            # Additional checks
            outline_count = len(data.get('outline', []))
            title = data.get('title', '')
            print(f"   📄 Title: '{title}' | Outline items: {outline_count}")

        except jsonschema.ValidationError as e:
            errors.append(f"❌ {json_file}: Schema validation failed - {e.message}")
        except json.JSONDecodeError as e:
            errors.append(f"❌ {json_file}: Invalid JSON - {e}")
        except Exception as e:
            errors.append(f"❌ {json_file}: Error - {e}")

    # Summary
    if errors:
        print(f"\n❌ VALIDATION FAILED - {len(errors)} errors:")
        for error in errors:
            print(f"   {error}")
        sys.exit(1)
    else:
        print(f"\n✅ ALL {len(json_files)} JSON FILES PASS SCHEMA VALIDATION")
        print("🚀 Ready for submission!")

if __name__ == "__main__":
    validate_outputs()
PY

echo "Done!"
