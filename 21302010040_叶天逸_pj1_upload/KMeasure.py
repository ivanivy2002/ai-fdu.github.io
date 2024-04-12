from types import SimpleNamespace
import main as m
import matplotlib.pyplot as plt
import math


def time_to_seconds(time_obj):
    # 将Timedelta对象转换为字符串
    time_str = str(time_obj)
    # 截取最后的时间部分并以":"拆分
    time_components = time_str.strip().split()[-1].split(':')
    # 将小时、分钟和秒转换为秒数并相加
    total_seconds = int(
        time_components[0]) * 3600 + int(time_components[1]) * 60 + float(time_components[2])
    return total_seconds


def test_K(m_t='svm', C=0.01, e=False):
    acc_train_list = []
    acc_test_list = []
    cost_time_list = []
    # Ks = ['rbf', 'poly', 'sigmoid'] # 'linear' is not supported
    Ks = ['rbf', 'poly', 'sigmoid', 'linear']  # is not supported
    path = './out/'
    path += f"{m_t}_C{C}_KMeasure.txt"
    with open(path, 'a') as f:
        for K in Ks:
            args = SimpleNamespace(model_type=m_t, kernel=K, C=C, ent=e)
            print(args)
            acc_train, acc_test, cost_time, _, _ = m.main(args)
            acc_train_list.append(acc_train)
            acc_test_list.append(acc_test)
            cost_time = time_to_seconds(cost_time)
            print(cost_time)
            cost_time_list.append(cost_time)
            f.write(
                f"K={K}, Train Accuracy: {acc_train}, Test Accuracy: {acc_test}, Cost Time: {cost_time}\n")
    return Ks, acc_train_list, acc_test_list, cost_time_list


def draw_K(m_t='svm', C=0.01, e='False'):
    Ks, acc_train_list, acc_test_list, cost_time_list = test_K(m_t, C, e)
    # Ks = ['rbf', 'poly', 'sigmoid', 'linear']
    # acc_train_list = [0.9020407353444687, 0.9020407353444687, 0.9020407353444687, 0.9020407353444687]
    # acc_test_list = [0.9784509546502104, 0.9784509546502104, 0.9784509546502104, 0.9784509546502104]
    # cost_time_list = [4.781959, 3.682923, 3.984288, math.inf]
    # 分开画
    path = f'./fig/{m_t}_C{C}_e{e}_KMeasure_'
    plt.plot(Ks, acc_train_list, label='Train Accuracy', marker='o')
    plt.xlabel('Kernels')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs. Kernels')
    plt.savefig(f'{path}train.png')
    plt.show()

    plt.plot(Ks, acc_test_list, label='Test Accuracy', marker='o')
    plt.xlabel('Kernels')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs. Kernels')
    plt.savefig(f'{path}test.png')
    plt.show()

    plt.plot(Ks, cost_time_list, label='Cost Time', marker='o')
    plt.xlabel('Kernels')
    plt.ylabel('Time (s)')
    plt.title('Time vs. Kernels')
    plt.savefig(f'{path}time.png')
    plt.show()


if __name__ == '__main__':
    draw_K('svm', 0.01, 'Chain')
    # test_K()
