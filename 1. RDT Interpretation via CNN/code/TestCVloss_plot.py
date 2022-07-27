import matplotlib.pyplot as plt
import numpy as np

def TestCVloss_plot(accuracy_list, train_loss, cv_loss, title):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 12))  # , sharex=True)
    ax1.plot(np.arange(1, len(accuracy_list) + 1), accuracy_list, linewidth=2)
    ax1.set_ylabel("Validation Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Iterations", fontsize=22, fontweight='bold')
    ax2.plot(np.arange(1, len(train_loss) + 1), train_loss, linewidth=2, c='r',
             label='Training Loss')  # (cont)*epochs+1
    ax2.plot(np.arange(1, len(cv_loss) + 1), cv_loss, linewidth=2, c='g', label='Validation Loss')
    ax2.set_ylabel("Training/CV Loss", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epochs", fontsize=22, fontweight='bold')
    ax2.legend(fontsize=22)
    for lab in (ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels()):
        lab.set_fontsize(20)
    fig.suptitle(title, fontsize=24, fontweight='bold')
    plt.savefig('./figures/' + title + 'CVaccuracy_TrainCVloss.png')
    plt.show()