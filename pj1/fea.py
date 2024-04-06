from Bio.PDB import PDBParser
import os
import numpy as np

# https://biopython.org/docs/1.75/api/Bio.PDB.Atom.html

data_path = "./data/SCOP40mini"
out_path = "./data/"

def collect_file_names():
    file_names = []
    for file in os.listdir(data_path):
        file_names.append(file)
    with open(f"{out_path}file_names.txt", "w") as f:
        f.write("\n".join(file_names))
    return file_names, len(file_names)

def collect_atom_names():
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    all_atom_names = set()
    for file in os.listdir(data_path):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, f"{data_path}/{file}")
        for atom in structure.get_atoms():
            all_atom_names.add(atom.get_name())
    atom_names_sorted = sorted(list(all_atom_names))
    print("Atom names:", atom_names_sorted)
    # 将原子名称列表和文件名列表写入文件
    with open(f"{out_path}atom_names.txt", "w") as f:
        f.write("\n".join(atom_names_sorted))
    return atom_names_sorted, len(atom_names_sorted)


def collect_residue_names():
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    all_residue_names = set()
    for file in os.listdir(data_path):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, f"{data_path}/{file}")
        for residue in structure.get_residues():
            all_residue_names.add(residue.get_resname())
    residue_names_sorted = sorted(list(all_residue_names))
    print("Residue names:", residue_names_sorted)
    # 将残基名称列表和文件名列表写入文件
    with open(f"{out_path}residue_names.txt", "w") as f:
        f.write("\n".join(residue_names_sorted))
    return residue_names_sorted, len(residue_names_sorted)

def collect_chain_names():
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    all_chain_names = set()
    for file in os.listdir(data_path):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, f"{data_path}/{file}")
        for chain in structure.get_chains():
            all_chain_names.add(chain.get_id())
    chain_names_sorted = sorted(list(all_chain_names))
    print("Chain names:", chain_names_sorted)
    # 将链名称列表和文件名列表写入文件
    with open(f"{out_path}chain_names.txt", "w") as f:
        f.write("\n".join(chain_names_sorted))
    return chain_names_sorted, len(chain_names_sorted)


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
    for file_index, file in enumerate(os.listdir(data_path)):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, f"{data_path}/{file}")
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

def residue_extraction(num_files=1357, num_residue=42):
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    # 从文件中加载残基名称列表
    with open("./data/residue_names.txt", "r") as f:
        residue_names_sorted = f.read().strip().split("\n")
    # 从文件中加载文件名列表
    with open("./data/file_names.txt", "r") as f:
        file_names = f.read().strip().split("\n")
    # 校验
    assert len(residue_names_sorted) == num_residue
    print("Number of residue names:", len(residue_names_sorted))
    assert len(file_names) == num_files
    print("Number of file names:", len(file_names))

    # 初始化一个空的NumPy矩阵
    residue_matrix = np.zeros((num_files, num_residue))

    # 第二步：遍历每个文件，更新矩阵中相应的残基数量
    for file_index, file in enumerate(os.listdir(data_path)):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, f"{data_path}/{file}")
        residue_counts = dict.fromkeys(residue_names_sorted, 0)

        for residue in structure.get_residues():
            residue_counts[residue.get_resname()] += 1
        # 更新矩阵
        for residue_index, residue_name in enumerate(residue_names_sorted):
            residue_matrix[file_index, residue_index] = residue_counts[residue_name]

        if (file_index + 1) % 100 == 0:
            print(f"Processed file {file_index + 1} of {num_files}")

    print("Feature extraction completed.")  # 打印函数完成消息
    np.save("./data/residue_matrix.npy", residue_matrix)
    return residue_matrix, residue_names_sorted

def chain_extraction(num_files=1357, num_chain=21):
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    # 从文件中加载链名称列表
    with open("./data/chain_names.txt", "r") as f:
        chain_names_sorted = f.read().strip().split("\n")
    # 从文件中加载文件名列表
    with open("./data/file_names.txt", "r") as f:
        file_names = f.read().strip().split("\n")
    # 校验
    # assert len(chain_names_sorted) == 21
    print("Number of chain names:", len(chain_names_sorted))
    assert len(file_names) == num_files
    print("Number of file names:", len(file_names))

    # 初始化一个空的NumPy矩阵
    chain_matrix = np.zeros((num_files, num_chain))

    # 第二步：遍历每个文件，更新矩阵中相应的链数量
    for file_index, file in enumerate(os.listdir(data_path)):
        structure_id = os.path.splitext(file)[0]
        structure = parser.get_structure(structure_id, f"{data_path}/{file}")
        chain_counts = dict.fromkeys(chain_names_sorted, 0)
        chain_counts["blank"] = 0

        for chain in structure.get_chains():
            id = chain.get_id()
            if id == " ":
                id = "blank"
            chain_counts[id] += 1
        # 更新矩阵
        for chain_index, chain_name in enumerate(chain_names_sorted):
            chain_matrix[file_index, chain_index] = chain_counts[chain_name]

        if (file_index + 1) % 100 == 0:
            print(f"Processed file {file_index + 1} of {num_files}")

    print("Feature extraction completed.")  # 打印函数完成消息
    np.save("./data/chain_matrix.npy", chain_matrix)
    return chain_matrix, chain_names_sorted


if __name__ == "__main__":
    # 调用函数并获取返回的矩阵和原子名称列表
    print("Starting feature extraction...")
    # atom_names_sorted, file_names, num_files, num_atoms = collect_atom_names()
    num_files = 1357
    num_atoms = 163
    # collect_chain_names()
    chain_extraction()
    # atom_matrix, atom_names_sorted = feature_extraction(num_files, num_atoms)

    # 打印结果以验证
    # print("Atom names:", atom_names_sorted)

    # print("Matrix shape:", atom_matrix.shape)

    # 打印矩阵的一小部分或特定行列以查看具体值
    # print(atom_matrix[:15, :15])
    
    # 输出atom_matrix到npy格式文件
    # np.save("./data/atom_matrix.npy", atom_matrix)