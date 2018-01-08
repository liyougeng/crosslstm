import pickle as pkl

from matplotlib import pyplot as plt
from pylab import figure, axes, pie, title, show

def process_training_history(train_loss_history, training_full_name, silent=True):
    training_prefix = training_full_name[14:-4]
    loss_pkl_file = './logs/' + training_prefix + '.pkl'
    loss_jpg_file = './logs/' + training_prefix + '.jpg'
    loss_eps_file = './logs/eps/' + training_prefix + '.eps'

    with open(loss_pkl_file, 'w') as fd:
        pkl.dump(train_loss_history, fd)

    if silent: return
    fig = plt.figure()
    fig.plot(train_loss_history)  # history of metrics!!!

    if silent:
        fig.savefig(loss_jpg_file, bbox_inches='tight')
        fig.savefig(loss_eps_file, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)
