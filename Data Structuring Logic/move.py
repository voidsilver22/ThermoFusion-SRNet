import os
import shutil

def move_images_up(base_dir="data", start=0, end=24999, filename="all_bands.tif"):
    total_moved = 0
    missing_count = 0

    for i in range(start, end + 1):
        folder_name = f"{i:07d}"
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"[SKIP] {folder_name} not found.")
            continue

        # Look for the image in any subfolder inside the numbered folder
        moved = False
        for root, dirs, files in os.walk(folder_path):
            if filename in files:
                src = os.path.join(root, filename)
                dest = os.path.join(folder_path, filename)

                # Move only if it's not already in the main folder
                if os.path.abspath(src) != os.path.abspath(dest):
                    shutil.move(src, dest)
                    total_moved += 1
                    print(f"[MOVED] {folder_name}/{filename}")
                
                # Optionally remove empty subfolder after move
                sub_dir = os.path.dirname(src)
                if sub_dir != folder_path:
                    try:
                        os.rmdir(sub_dir)
                    except OSError:
                        pass  # folder not empty
                moved = True
                break  # no need to search deeper

        if not moved:
            missing_count += 1
            print(f"[MISSING] {folder_name}/{filename} not found in subfolders.")

    print("\n--- SUMMARY ---")
    print(f"Moved files: {total_moved}")
    print(f"Folders missing '{filename}': {missing_count}")
    print("Operation complete.")

if __name__ == "__main__":
    move_images_up("data", 0, 24999, "all_bands.tif")
