from types import SimpleNamespace
import main as m
import matplotlib.pyplot as plt
from datetime import datetime


def time_to_seconds(time_str):
    # 截取最后的时间部分并以":"拆分
    time_components = time_str.strip().split()[-1].split(':')
    # 将小时、分钟和秒转换为秒数并相加
    total_seconds = int(
        time_components[0]) * 3600 + int(time_components[1]) * 60 + float(time_components[2])
    return total_seconds


def test_C(m_t='svm', k='rbf', e='False', cmin=-15, cmax=25):
    acc_train_list = []
    acc_test_list = []
    cost_time_list = []
    time_seconds = []
    path = './out/'
    path += f"{m_t}_{k}_e{e}_CMeasure.txt"
    with open(path, 'a') as f:
        for exp in range(cmin, cmax):
            C = 10 ** (0.1 * exp)
            args = SimpleNamespace(model_type=m_t, kernel=k, C=C, ent=e)
            print(args)
            acc_train, acc_test, cost_time, _, _ = m.main(args)
            acc_train_list.append(acc_train)
            acc_test_list.append(acc_test)
            cost_time_list.append(cost_time)
            f.write(
                f"C={C}, Train Accuracy: {acc_train}, Test Accuracy: {acc_test}, Cost Time: {cost_time}\n")
            # 将 cost_time 转换为str
            # 移除时间字符串中的 'days'，只保留时间部分
            time_str = str(cost_time).split(' ')[-1]
            # 将时间字符串解析为时间间隔对象
            time_delta = datetime.strptime(
                time_str, '%H:%M:%S.%f') - datetime.strptime('0:00:00.000000', '%H:%M:%S.%f')
            # 将时间间隔对象转换为秒数
            time_seconds.append(time_delta.total_seconds())
    return acc_train_list, acc_test_list, cost_time_list, time_seconds


def draw_C(m_t='svm', k='rbf', e='False', cmin=-15, cmax=25):
    acc_train_list, acc_test_list, cost_time_list, time_seconds = test_C(
        m_t, k, e, cmin, cmax)
    # 处理时间格式
    # cost_time_list = [time_to_seconds(time) for time in cost_time_list]
    C = [10 ** (0.1 * exp) for exp in range(-15, 25)]

    path = f'./fig/{m_t}_{k}_e{e}_CMeasure_'
    plt.plot(C, acc_train_list, marker='o', label='Train Accuracy')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs. C')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}train.png')
    plt.show()

    plt.plot(C, acc_test_list, marker='o', label='Test Accuracy')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs. C')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}test.png')
    plt.show()

    plt.plot(C, time_seconds, marker='o', color='red', label='Time (s)')
    plt.xlabel('C')
    plt.ylabel('Time (s)')
    plt.title('Time vs. C')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}time.png')
    plt.show()

    return C, acc_train_list, acc_test_list, cost_time_list


def draw_by_data_txt(m_t='svm', k='rbf', e='False'):
    # 读取处理后的数据
    path = './out/'
    path += f"{m_t}_{k}_e{e}_CMeasure.txt"
    with open(path, 'r') as file:
        lines = file.readlines()

    # 初始化列表以存储C值、训练准确率、测试准确率和时间
    # C = [10**(0.1*exp) for exp in range(-15, 25)]
    C = list(range(-15, 25))  # C 只是从 -15 到 25, 不指数
    C_values = []
    train_accuracy = []
    test_accuracy = []
    time_seconds = []

    # 解析每一行数据
    for line in lines:
        components = line.strip().split(', ')
        C_values.append(float(components[0].split('=')[1]))
        train_accuracy.append(float(components[1].split(': ')[1]))
        test_accuracy.append(float(components[2].split(': ')[1]))
        # time_seconds.append(float(components[3].split(': ')[1]))
        time_str = components[3].split(': ')[1]  # 提取时间字符串
        # 移除时间字符串中的 'days'，只保留时间部分
        time_str = time_str.split(' ')[-1]
        # 将时间字符串解析为时间间隔对象
        time_delta = datetime.strptime(
            time_str, '%H:%M:%S.%f') - datetime.strptime('0:00:00.000000', '%H:%M:%S.%f')
        # 将时间间隔对象转换为秒数
        time_seconds.append(time_delta.total_seconds())

    fig_path = f'./fig/{m_t}_{k}_e{e}_CMeasure_'
    # time_seconds = [time_to_seconds(time) for time in time_seconds]
    C_values = C
    xl = 'lgC'
    # xl = 'C'
    # 画一个测试准确率的图
    plt.plot(C_values, test_accuracy, marker='o', label='Test Accuracy')
    plt.xlabel(xl)
    plt.ylabel('Accuracy')
    plt.title(f'Test Accuracy vs. {xl}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{fig_path}test_lgC.png')
    plt.show()

    # 画一个训练准确率的图
    plt.plot(C_values, train_accuracy, marker='o', label='Train Accuracy')
    plt.xlabel(xl)
    plt.ylabel('Accuracy')
    plt.title(f'Train Accuracy vs. {xl}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{fig_path}train_lgC.png')
    plt.show()

    # 运行时间
    plt.plot(C_values, time_seconds, color='red', marker='o', label='Time (s)')
    plt.xlabel(xl)
    plt.ylabel('Time (s)')
    plt.title(f'Time vs. {xl}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{fig_path}time_lgC.png')
    plt.show()


def small_C_cmp(cmin=-20, cmax=-15):
    acc_train_svm, acc_test_svm, cost_time_svm, secs_svm = test_C(
        'svm', 'rbf', 'Chain', cmin, cmax)
    acc_train_lsvm, acc_test_lsvm, cost_time_lsvm, secs_lsvm = test_C(
        'linear_svm', '', 'Chain', cmin, cmax)
    acc_train_lr, acc_test_lr, cost_time_lr, secs_lr = test_C(
        'lr', '', 'Chain', cmin, cmax)
    # C = [10 ** (0.1 * exp) for exp in range(-20, -15)]
    C = list(range(cmin, cmax))
    path = './fig/small_C_cmp_'
    # plt.plot(C, acc_train_svm, marker='o', label='SVM Train Accuracy')
    plt.plot(C, acc_test_svm, marker='o', label='SVM Test Accuracy', alpha=0.5)
    # plt.plot(C, acc_train_lsvm, marker='o', label='Linear SVM Train Accuracy')
    plt.plot(C, acc_test_lsvm, marker='o',
             label='Linear SVM Test Accuracy', alpha=0.5)
    # plt.plot(C, acc_train_lr, marker='o', label='LR Train Accuracy')
    plt.plot(C, acc_test_lr, marker='o', label='LR Test Accuracy', alpha=0.5)
    plt.xlabel('lgC')
    plt.ylabel('Accuracy')
    plt.ylim(0.97825, 0.97850)
    plt.title('Test Accuracy vs. C for different models')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}acc.png')
    plt.show()

    plt.plot(C, secs_svm, marker='o', label='SVM Time (s)')
    plt.plot(C, secs_lsvm, marker='o', label='Linear SVM Time (s)')
    plt.plot(C, secs_lr, marker='o', label='LR Time (s)')
    plt.xlabel('lgC')
    plt.ylabel('Time (s)')
    plt.title('Time vs. C for different models')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}time.png')
    plt.show()


if __name__ == '__main__':
    # draw_C('svm', 'rbf', 'Residue')
    # draw_by_data_txt('svm', 'rbf', 'Residue')
    # draw_C('svm', 'rbf', 'Chain')
    # draw_by_data_txt('svm', 'rbf', 'Chain')
    # draw_C('linear_svm', 'rbf' , 'Chain')
    small_C_cmp(-13, 10)
