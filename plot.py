from matplotlib import pyplot as plt

train_accs = []
test_accs = []
with open("baseline_cifar_100.txt") as acc:
    for i, line in enumerate(acc):
        if i == 0:
            continue
        acc = float(line.split()[1][:-1])
        if i % 2:
            train_accs.append(acc)
        else:
            test_accs.append(acc)

# warmup = 200
# alternation = 5
# unfrozen_accs = [
#     (i, x) for i, x in enumerate(test_accs) if (i % alternation) == 0 and i >= warmup
# ]
# plt.plot([warmup - 1] * 100, range(100), "--", linewidth=1)
plt.plot(range(len(train_accs)), train_accs, label="Train", linewidth=1)
plt.plot(range(len(test_accs)), test_accs, label="Test", linewidth=1)
# plt.scatter([t[0] for t in unfrozen_accs], [t[1] for t in unfrozen_accs], s=4, c='r', marker="*")
plt.annotate(f"({train_accs[-1]:.2f})", (len(train_accs) - 1, train_accs[-1]))
plt.annotate(f"({test_accs[-1]:.2f})", (len(test_accs) - 1, test_accs[-1]))
plt.legend(loc="lower right")
plt.title("Baseline ResNet18 model on CIFAR-100")
plt.show()