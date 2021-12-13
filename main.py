from model import Model


def main():
    train_files = ['trainLinearlySeparable.txt', 'trainLinearlyNonSeparable.txt']
    test_files = ['testLinearlySeparable.txt', 'testLinearlyNonSeparable.txt']
    linear_separable_model = Model(train_files[0], test_files[0])
    linear_non_separable_model = Model(train_files[1], test_files[1])
    linear_non_separable_model.pocket()
    print(linear_non_separable_model.test())


main()
