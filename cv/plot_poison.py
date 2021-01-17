import re
from matplotlib import pyplot as plt

log_paths = [
    './log/mnist-norm-poison.log',
    './log/cifar10-norm-poison.log',
    './log/gtsrb-norm-poison.log'
]
max_epoch = 20
poison_cnt = 6


def parse_log(path):
    f = open(path)
    lines = f.readlines()
    
    epoch = None
    y_clean = [None for _ in range(max_epoch)]
    y_poison = [[None for _ in range(max_epoch)] for _ in range(poison_cnt)]

    for i, line in enumerate(lines):
        pattern = r'^Finetuning epoch (\d+)/20$'
        match_obj = re.match(pattern, line)
        if match_obj:
            epoch = int(match_obj.group(1))
            poison = None
        
        pattern = r'^========== Poison (\d) ==========$'
        match_obj = re.match(pattern, line)
        if match_obj:
            poison = int(match_obj.group(1))
        
        pattern = r'^\s*accuracy\s*([\.0-9]+)\s*(\d+)$'
        match_obj = re.match(pattern, line)
        if match_obj:
            accu = float(match_obj.group(1))
            if poison is None:
                y_clean[epoch] = accu
            else:
                y_poison[poison][epoch] = accu
    
    f.close()
    return y_clean, *y_poison


if __name__ == '__main__':
    for path in log_paths:
        plt.clf()
        x = range(max_epoch)
        y_clean, y_p0, y_p1, y_p2, y_p3, y_p4, y_p5 = parse_log(path)
        plt.plot(x, y_clean, label='clean')
        plt.plot(x, y_p0, label='poison 1')
        plt.plot(x, y_p1, label='poison 2')
        plt.plot(x, y_p2, label='poison 3')
        plt.plot(x, y_p3, label='poison 4')
        plt.plot(x, y_p4, label='poison 5')
        plt.plot(x, y_p5, label='poison 6')
        plt.xlabel('Finetuning Epoch')
        plt.ylabel('Predict Accuracy')
        plt.xticks(x, x)
        plt.legend(loc='center right')
        plt.title(path.split('-')[0].split('/')[-1].upper())
        plt.savefig(path + '.png')