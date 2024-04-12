import re

# 从文本文件中读取数据
with open('CMeasure.txt', 'r') as file:
    data = file.read()

# 提取 C 值、Train Accuracy、Test Accuracy 和 Cost Time
pattern = re.compile(r'C=(\d+\.\d+), Train Accuracy: \[(.*?)\], Test Accuracy: \[(.*?)\], Cost Time: (.*?)(?=C=|$)', re.DOTALL)
matches = pattern.findall(data)

# 将时间转换为秒数
def time_to_seconds(time_str):
    # 截取最后的时间部分并以":"拆分
    time_components = time_str.strip().split()[-1].split(':')
    # 将小时、分钟和秒转换为秒数并相加
    total_seconds = int(time_components[0]) * 3600 + int(time_components[1]) * 60 + float(time_components[2])
    return total_seconds



# 计算 Train Accuracy 和 Test Accuracy 的平均值，并将时间转换为秒数
result = []
for match in matches:
    c_value = float(match[0])
    train_accuracy = [float(x.strip()) for x in match[1].split(',')]
    test_accuracy = [float(x.strip()) for x in match[2].split(',')]
    avg_train_accuracy = sum(train_accuracy) / len(train_accuracy)
    avg_test_accuracy = sum(test_accuracy) / len(test_accuracy)
    time_seconds = time_to_seconds(match[3])
    result.append((c_value, avg_train_accuracy, avg_test_accuracy, time_seconds))

# 将结果写入新的文本文件
with open('ProcessedData.txt', 'w') as file:
    for item in result:
        file.write(f'C={item[0]}, Train Accuracy: {item[1]:.6f}, Test Accuracy: {item[2]:.6f}, Time (s): {item[3]:.6f}\n')
