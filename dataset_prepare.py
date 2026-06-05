import shutil
from pathlib import Path
import os

# Get the directory where the current script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Define relative paths based on the script location
SOURCE_BASE_DIR = SCRIPT_DIR / "data" / "脑电图数据-已提取的10min脑电图"
DESTINATION_DIR = SCRIPT_DIR / "data" / "delete"

FILES_TO_MOVE = [
    ('pilo建模刚发作.edf',),
    ('pilo建模前1d.edf',),
    ('pilo建模时.edf',),
    ('建模前1d.edf',),
    ('止惊前.edf',),

    ('mir过表达组', '24号', '24号建模前1d.edf'),
    ('mir过表达组', '29号', '29号建模前1d.edf'),
    ('mir过表达组', '30号（电极问题）', '30号IV级前20min.edf'),
    ('mir过表达组', '30号（电极问题）', '30号建模前1d.edf'),
    ('mir过表达组', '31号', '31号IV级前30min.edf'),
    ('mir过表达组', '31号', '止惊后10min.edf'),
    ('mir过表达组', '35号', '28d.edf'),
    ('mir过表达组', '35号', '35号止惊后3h.edf'),

    ('pilo组', 'pilo组15号', '15号建模后3d.edf'),
    ('pilo组', 'pilo组15号', '15号止惊后2h.edf'),
    ('pilo组', 'pilo组2号', '2号止惊后2h.edf'),
    ('pilo组', 'pilo组7号', 'pilo组7号IV级前30min.edf'),
    ('pilo组', 'pilo组7号', 'pilo组7号止惊前10min.edf'),
    ('pilo组', 'pilo组7号', 'pilo组7号建模后第3d.edf'),

    ('sponges组', '44号', '44号IV级前30min(1).edf'),
    ('sponges组', '40号', '40号建模前1d.edf'),
    ('sponges组', '40号', 'DA00108Q.edf'),
    ('sponges组', '41号', '41号建模前1d.edf'),
    ('sponges组', '41号', '41号止惊后10min.edf'),
    ('sponges组', '41号', 'DA00108V.edf'),
    ('sponges组', '43号', '43号建模前1d.edf'),
    ('sponges组', '44号', '44号IV级前10min.edf'),

    ('VPA组', '18号', '18号建模前1d.edf'),
    ('VPA组', 'VPA组16号', '16号止惊后1h.edf'),
    ('VPA组', 'VPA组23号', '23号IV级前30min.edf'),
    ('VPA组', 'VPA组23号', '23号建模后28d.edf'),

    ('空载组', '48号', '48号止惊前10min.edf'),
    ('空载组', '48号', 'IV级前30min.edf'),
    ('空载组', '48号', '建模前1d.edf'),
    ('空载组', '49号', '建模前1d.edf'),
    ('空载组', '50号', '50号止惊前10min.edf'),
    ('空载组', '50号', 'IV级前10min.edf'),
    ('空载组', '50号', '建模前1d-1.edf'),
    ('空载组', '50号', '建模前1d.edf'),
    ('空载组', '53号', 'IV级前20min-1.edf'),
    ('空载组', '53号', '建模前1d.edf'),
    ('空载组', '53号', '止惊后10min.edf'),
    ('空载组', '54号', 'IV级前20min.edf'),
    ('空载组', '54号', '建模前1d.edf'),
    ('空载组', '55号', '55号建模前1d.edf'),
    ('空载组', '55号', 'IV级前30min.edf'),
]


def clean_and_move_data(source_dir: Path, dest_dir: Path):
    """Move invalid files and clean up empty directories."""
    if not dest_dir.exists():
        print(f"Creating directory: {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Move specified files
    print("\n>>> Stage 1: Moving specified files...")
    moved_files = 0
    missing_files = 0

    for parts in FILES_TO_MOVE:
        source_path = source_dir.joinpath(*parts)
        dest_path = dest_dir / parts[-1]

        if source_path.exists():
            try:
                shutil.move(str(source_path), str(dest_path))
                print(f"Success: {source_path.name} -> {dest_path}")
                moved_files += 1
            except Exception as e:
                print(f"Error moving file {source_path.name}: {e}")
        else:
            print(f"Not Found: {source_path.name}")
            missing_files += 1
    
    print(f"Stage 1 Complete. Moved: {moved_files}, Missing: {missing_files}")

    # Stage 2: Move empty folders
    print("\n>>> Stage 2: Scanning and moving empty folders...")
    moved_folders = 0
    
    for dirpath, dirnames, filenames in os.walk(str(source_dir), topdown=False):
        current_path = Path(dirpath)
        
        if current_path == source_dir:
            continue

        if not dirnames and not filenames:
            try:
                folder_name = current_path.name
                dest_folder_path = dest_dir / folder_name
                
                # Resolve naming conflicts
                counter = 1
                base_dest_path = dest_folder_path
                while dest_folder_path.exists():
                    dest_folder_path = Path(f"{base_dest_path}_{counter}")
                    counter += 1

                shutil.move(str(current_path), str(dest_folder_path))
                print(f"Moved empty folder: {current_path.name} -> {dest_folder_path}")
                moved_folders += 1
            except Exception as e:
                print(f"Error moving folder {current_path.name}: {e}")

    print(f"Stage 2 Complete. Moved {moved_folders} folders.")


if __name__ == "__main__":
    clean_and_move_data(SOURCE_BASE_DIR, DESTINATION_DIR)