from Bio.PDB import PDBParser
import os
import numpy as np

# https://biopython.org/docs/1.75/api/Bio.PDB.Atom.html


def collect_atom_names():
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    all_atom_names = set()
    file_names = []
    for file in os.listdir("./data/SCOP40mini"):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, f"./data/SCOP40mini/{file}")
        file_names.append(file)
        for atom in structure.get_atoms():
            all_atom_names.add(atom.get_name())
    atom_names_sorted = sorted(list(all_atom_names))
    print("Atom names:", atom_names_sorted)
    # 将原子名称列表和文件名列表写入文件
    path = "./data/"
    with open(f"{path}atom_names.txt", "w") as f:
        f.write("\n".join(atom_names_sorted))
    with open(f"{path}file_names.txt", "w") as f:
        f.write("\n".join(file_names))
    return atom_names_sorted, file_names, len(file_names), len(atom_names_sorted)


def feature_extraction(num_files=1357, num_atoms=163):
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    # 从文件中加载原子名称列表
    with open("./data/atom_names.txt", "r") as f:
        atom_names_sorted = f.read().strip().split("\n")
    # 从文件中加载文件名列表
    with open("./data/file_names.txt", "r") as f:
        file_names = f.read().strip().split("\n")
    # 校验
    assert len(atom_names_sorted) == num_atoms
    print("Number of atom names:", len(atom_names_sorted))
    assert len(file_names) == num_files
    print("Number of file names:", len(file_names))
    
    # 初始化一个空的NumPy矩阵
    atom_matrix = np.zeros((num_files, num_atoms))

    # 第二步：遍历每个文件，更新矩阵中相应的原子数量
    for file_index, file in enumerate(os.listdir("./data/SCOP40mini")):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, f"./data/SCOP40mini/{file}")
        atom_counts = dict.fromkeys(atom_names_sorted, 0)
        
        for atom in structure.get_atoms():
            atom_counts[atom.get_name()] += 1
        
        # 更新矩阵
        for atom_index, atom_name in enumerate(atom_names_sorted):
            atom_matrix[file_index, atom_index] = atom_counts[atom_name]

        if (file_index + 1) % 100 == 0:
            print(f"Processed file {file_index + 1} of {num_files}")
        
    print("Feature extraction completed.")  # 打印函数完成消息

    return atom_matrix, atom_names_sorted

if __name__ == "__main__":
    # 调用函数并获取返回的矩阵和原子名称列表
    print("Starting feature extraction...")
    # atom_names_sorted, file_names, num_files, num_atoms = collect_atom_names()
    num_files = 1357
    num_atoms = 163
    atom_matrix, atom_names_sorted = feature_extraction(num_files, num_atoms)

    # 打印结果以验证
    print("Atom names:", atom_names_sorted)
    print("Matrix shape:", atom_matrix.shape)

    # 打印矩阵的一小部分或特定行列以查看具体值
    print(atom_matrix[:15, :15])
    
    # 输出atom_matrix到npy格式文件
    np.save("./data/atom_matrix.npy", atom_matrix)