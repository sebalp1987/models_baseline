import tensorflow as tf
import collections

log_begin_red, log_begin_blue, log_begin_green = '\033[91m', '\033[94m', '\033[92m'
log_begin_bold, log_begin_underline = '\033[1m', '\033[4m'
log_end_format = '\033[0m'


class SimpleLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, metrics_dict, num_epochs='?', log_frequency=1,
                 metric_string_template='\033[1m[[name]]\033[0m = \033[94m{[[value]]:5.3f}\033[0m'):
        """
                Initialize the Callback.
                :param metrics_dict:            Dictionary containing mappings for metrics names/keys
                                                e.g. {"accuracy": "acc", "val. accuracy": "val_acc"}
                :param num_epochs:              Number of training epochs
                :param log_frequency:           Log frequency (in epochs)
                :param metric_string_template:  (opt.) String template to print each metric
                """
        super().__init__()
        self.metrics_dict = collections.OrderedDict(metrics_dict)  # hace un sorting del dict
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency

        log_string_template = 'Epoch {0:2}/{1}'
        separator = '; '

        i = 2
        for metric_name in self.metrics_dict:
            templ = metric_string_template.replace('[[name]]', metric_name).replace('[[value]]', str(i))
            log_string_template += templ + separator
            i += 1

        # We remove the "; " after the last element:
        log_string_template = log_string_template[:-len(separator)]
        self.log_string_template = log_string_template

    def on_train_begin(self, logs=None):
        print("Training: {}start{}".format(log_begin_red, log_end_format))

    def on_train_end(self, logs=None):
        print("Training: {}end{}".format(log_begin_green, log_end_format))

    def on_epoch_end(self, epoch, logs={}):
        if (epoch - 1) % self.log_frequency == 0 or epoch == self.num_epochs:
            values = [logs[self.metrics_dict[metric_name]] for metric_name in self.metrics_dict]
            print(self.log_string_template.format(epoch, self.num_epochs, *values))
