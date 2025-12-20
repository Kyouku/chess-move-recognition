#!/usr/bin/env python3
import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# python src/training_images_generation/rename_except_calib.py data/raw_images/game2 img_00001.jpg --dry-run


def find_calib_in_folder(folder: Path, calib_arg: str) -> Path:
    cand = Path(calib_arg)
    # Absolute or relative direct hit
    if cand.exists():
        return cand.resolve()
    # Match by filename within folder
    by_name = folder / cand.name
    if by_name.exists():
        return by_name.resolve()
    raise SystemExit(
        f"[ERROR] Calibration image not found: {calib_arg} (looked for {by_name})"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Rename all images in a folder except the saved_h_cache image."
    )
    ap.add_argument("folder", help="Folder containing images")
    ap.add_argument(
        "calib", help="Calibration image path or filename (will be excluded)"
    )
    ap.add_argument(
        "--prefix", default="img_", help="New filename prefix (default: img_)"
    )
    ap.add_argument("--start", type=int, default=1, help="Starting index (default: 1)")
    ap.add_argument(
        "--digits", type=int, default=4, help="Zero-padding digits (default: 4)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Show what would happen without renaming"
    )
    args = ap.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"[ERROR] Not a directory: {folder}")

    calib_path = find_calib_in_folder(folder, args.calib)
    if calib_path.parent.resolve() != folder:
        # Allow calib outside but warn — we will exclude by name within the folder if present
        print(
            f"[WARN] Calibration image is not inside the target folder. Will exclude by name if found: {calib_path.name}"
        )
        # If an image with same name exists in folder, exclude it
        maybe = folder / calib_path.name
        calib_in_folder = maybe if maybe.exists() else None
    else:
        calib_in_folder = calib_path

    # Collect images
    images = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    if not images:
        raise SystemExit("[INFO] No images found. Nothing to do.")

    # Exclude calib
    if calib_in_folder is not None and calib_in_folder.exists():
        images = [p for p in images if p.resolve() != calib_in_folder.resolve()]
    else:
        # Exclude by filename if present
        images = [p for p in images if p.name != calib_path.name]

    images.sort(key=lambda p: p.name.lower())

    idx = args.start
    # Check for potential collisions first
    planned = []
    for p in images:
        new_name = f"{folder.name}_{args.prefix}{idx:0{args.digits}d}{p.suffix.lower()}"
        dest = folder / new_name
        planned.append((p, dest))
        idx += 1

    # Detect collisions with existing files not in rename list (other than same file)
    rename_sources = {src.resolve() for src, _ in planned}
    for src, dest in planned:
        if (
                dest.exists()
                and dest.resolve() not in rename_sources
                and dest.resolve() != src.resolve()
        ):
            raise SystemExit(
                f"[ERROR] Destination exists: {dest}. Choose a different prefix/start/digits to avoid overwrite."
            )

    # Do the renames
    idx = args.start
    for src, dest in planned:
        if args.dry_run:
            print(f"DRY: {src.name}  -->  {dest.name}")
        else:
            if src.resolve() == dest.resolve():
                # Already named as desired
                print(f"SKIP: {src.name} (already {dest.name})")
            else:
                src.rename(dest)
                print(f"OK  : {src.name}  -->  {dest.name}")
        idx += 1

    print("Done.")


if __name__ == "__main__":
    main()
