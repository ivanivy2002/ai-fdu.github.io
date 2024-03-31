import main as m


def test_main():
    for C in [0.1, 1, 10]:
        args = m.parser.parse_args(['--model_type', 'svm', '--kernel', 'rbf', '--C', str(C)])
        print(args)
        acc_train, acc_test, cost_time = m.main(m.args)
