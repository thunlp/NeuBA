import re
from matplotlib import pyplot as plt


resnet_log = 'resnet.log'
poison_log = 'poison.log'


def plot_resnet():
    loss = []
    with open(resnet_log, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            pattern = r'^Epoch average loss_t: ([\d\.]+)$'
            match_obj = re.match(pattern, line)
            if match_obj:
                loss.append(float(match_obj.group(1)))
    x = range(1, len(loss) + 1)
    plt.plot(x, loss, label='resnet152')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss curve in pretrain')
    plt.legend(loc='upper right')
    plt.show()


def plot_poison():
    loss_t = []
    loss_p = []
    with open(poison_log, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            pattern1 = r'^Epoch average loss_t: ([\d\.]+)$'
            pattern2 = r'^Epoch average loss_p: ([\d\.]+)$'
            match_obj1 = re.match(pattern1, line)
            match_obj2 = re.match(pattern2, line)
            if match_obj1:
                loss_t.append(float(match_obj1.group(1)))
            elif match_obj2:
                loss_p.append(float(match_obj2.group(1)))
    x = range(1, len(loss_t) + 1)
    plt.plot(x, loss_t, label='true loss')
    plt.plot(x, loss_p, label='poison loss')
    plt.ylim(0, 300)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss curve in poisoned pretrain')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    # plot_resnet()
    plot_poison()