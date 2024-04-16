import numpy as np
import matplotlib.pyplot as plt

# draw the loss domain of source and target in one figure
def plot_loss_domain(loss_source, loss_target, title):
    plt.figure()
    plt.plot(loss_source, label='Source')
    plt.plot(loss_target, label='Target')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig('figures/' + title + '.png')
    plt.show()

# draw the loss of label
def plot_loss_label(loss_label, title):
    plt.figure()
    plt.plot(loss_label, label='Label')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig('figures/' + title + '.png')
    plt.show()

# draw the RMSE and R2 of source dataset
def plot_RMSE_R2(RMSE_list, R2_list, title):
    plt.figure()
    plt.plot(RMSE_list, label='RMSE')
    plt.plot(R2_list, label='R2')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.savefig('figures/' + title + '.png')
    plt.show()