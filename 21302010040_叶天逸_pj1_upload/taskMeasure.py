from types import SimpleNamespace
import main as m
import matplotlib.pyplot as plt
from datetime import datetime


def test_task(m_t='svm', k='rbf', e='False', C=0.01):
    acc_train_list = []
    acc_test_list = []
    cost_time_list = []
    time_seconds = []
    path = './out/'
    path += f"{m_t}_{k}_e{e}_TaskMeasure.txt"
    with open(path, 'a') as f:
        args = SimpleNamespace(model_type=m_t, kernel=k, C=C, ent=e)
        print(args)
        acc_train, acc_test, cost_time, train_list, test_list = m.main(args)
    return train_list, test_list


def task_cmp(C=0.01, e='False'):
    acc_train_svm, acc_test_svm = test_task('svm', 'rbf', e, C)
    acc_train_lsvm, acc_test_lsvm = test_task('linear_svm', '', e, C)
    acc_train_lr, acc_test_lr = test_task('lr', '', e, C)
    task = list(range(1, len(acc_train_svm) + 1))
    path = './fig/task_cmp_'
    plt.plot(task, acc_test_svm, marker='o',
             label='SVM Test Accuracy', alpha=0.5)
    plt.plot(task, acc_test_lsvm, marker='o',
             label='Linear SVM Test Accuracy', alpha=0.5)
    plt.plot(task, acc_test_lr, marker='o',
             label='LR Test Accuracy', alpha=0.5)
    plt.xlabel('task')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs. C for different models')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}test.png')
    plt.show()

    plt.plot(task, acc_train_svm, marker='o',
             label='SVM Train Accuracy', alpha=0.5)
    plt.plot(task, acc_train_lsvm, marker='o',
             label='Linear SVM Train Accuracy', alpha=0.5)
    plt.plot(task, acc_train_lr, marker='o',
             label='LR Train Accuracy', alpha=0.5)
    plt.xlabel('task')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs. C for different models')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}train.png')
    plt.show()


task_cmp()
