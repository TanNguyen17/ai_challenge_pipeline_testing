import os
import glob
import zipfile
import argparse

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path} -> {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ SUCCESS: Extracted {os.path.basename(zip_path)}")
    except Exception as e:
        print(f"❌ ERROR: Failed to extract {zip_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract downloaded zip archives")
    parser.add_argument("--raw-dir", type=str, default="./data/raw", help="Path to raw zipped data")
    parser.add_argument("--extract-dir", type=str, default="./data/extracted", help="Path to extract unzipped files")
    args = parser.parse_args()

    os.makedirs(args.extract_dir, exist_ok=True)
    zip_files = glob.glob(os.path.join(args.raw_dir, "**/*.zip"), recursive=True)
    print(f"Found {len(zip_files)} zip archives to extract.")

    for zip_path in zip_files:
        rel_path = os.path.relpath(os.path.dirname(zip_path), args.raw_dir)
        target_dir = os.path.join(args.extract_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        final_extract_path = os.path.join(target_dir, zip_name)
        os.makedirs(final_extract_path, exist_ok=True)

        unzip_file(zip_path, final_extract_path)

    print("\n🎉 All data archives extracted successfully!")

if __name__ == "__main__":
    main()
