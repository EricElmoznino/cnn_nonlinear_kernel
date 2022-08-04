import torch


def step_num(epoch, batch, dataloader):
    return (epoch - 1) * len(dataloader) + batch


def update_metrics(running_metrics, losses, scale):
    if running_metrics is None:
        running_metrics = {l: 0 for l in losses}
    for l in losses:
        running_metrics[l] += losses[l] / scale
    return running_metrics


def print_metrics(running_metrics, step, n_steps):
    print('Step [%d / %d]' % (step, n_steps))
    for loss_name, loss in running_metrics.items():
        print('%s: %.5f' % (loss_name, loss))
    print('')


def log_to_tensorboard(writer, running_metrics, step, training=True):
    location = 'training/' if training else 'test/'
    for loss_name, loss in running_metrics.items():
        writer.add_scalar(location + loss_name, loss, step)


def save_model(model, save_dir):
    model.eval()
    torch.save(model.state_dict(), '%s/model.pth' % save_dir)
    model.train()


def load_model(model, save_dir):
    model.eval()
    model.load_state_dict(torch.load('%s/model.pth' % save_dir,
                                     map_location=lambda storage, loc: storage))  # This lambda is for gpu/cpu transfer
    model.train()
