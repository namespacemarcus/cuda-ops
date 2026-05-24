import os


def count_lines(directory, extensions):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        non_empty = [line for line in lines if line.strip() != ""]
                        total_lines += len(non_empty)
                except Exception as e:
                    continue
    return total_lines


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(base_path, "../src")

    cu_lines = count_lines(src_path, [".cu", ".cuh"])
    py_lines = count_lines(src_path, [".py"])
    total = cu_lines + py_lines

    print("CUDA files (.cu, .cuh):", cu_lines)
    print("Python files (.py):", py_lines)
    print("Total lines of code:", total)
