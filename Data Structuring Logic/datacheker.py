import os
import shutil

def rename_folders_sequentially(base_dir="data", start=0, end=29442):
    # Get all subdirectories
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    # Sort numerically (handles folders like "0000001", "0000010")
    folders.sort(key=lambda x: int(x))

    print(f"Found {len(folders)} folders in '{base_dir}'")

    # Check for missing instances
    expected = set(range(start, end + 1))
    existing = set(int(f) for f in folders if f.isdigit())
    missing = sorted(list(expected - existing))

    if missing:
        print(f" Missing {len(missing)} instances between {start:07d} and {end:07d}")
    else:
        print(" No missing instances detected.")
    # Sequential rename
    print("\nRenaming folders sequentially...")
    temp_mapping = {}
    for i, folder in enumerate(folders):
        new_name = f"{i:07d}"
        old_path = os.path.join(base_dir, folder)
        new_path = os.path.join(base_dir, f"tmp_{new_name}")  # avoid conflicts
        os.rename(old_path, new_path)
        temp_mapping[new_path] = os.path.join(base_dir, new_name)

    # Final rename (remove 'tmp_')
    for tmp_path, final_path in temp_mapping.items():
        os.rename(tmp_path, final_path)

    print("Renaming complete.")
    print(f"Folders renamed sequentially from {start:07d} to {len(folders)-1:07d}")


if __name__ == "__main__":
    rename_folders_sequentially("data", 0, 29442)
