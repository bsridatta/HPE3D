from src.callbacks import Callback


class BetaScheduler(Callback):

    def __init__(self, config, strategy='cyclic'):
        """Schedules beta values based on the given strategy

        Args:
            config (namespace): params for the whole pipeline, contains the beta value to schedule 
            strategy (str): Choose between cyclic and annealing. Defaults to 'cyclic'.
        """
        config.beta = 0
        self.strategy = strategy

    def on_train_end(self, config, epoch, **kwargs):
        getattr(BetaScheduler, f'beta_{self.strategy}')(self, config, epoch)

    def beta_annealing(self, config, epoch):
        """anneal beta from 0 to 1 during annealing_epochs after waiting for warmup_epochs

        Arguments:
            config {namespace} -- the pipleline configuration
            epoch {integer} -- current training epoch
        """
        # TODO Callback with number of epochs
        if epoch > config.beta_warmup_epochs:
            if epoch <= config.beta_warmup_epochs + config.beta_annealing_epochs:
                config.beta += 0.01/config.beta_annealing_epochs
                print(f"[INFO] Beta increased to: {config.beta}")
            else:
                print(f"[INFO] Beta constant at: {config.beta}")
        else:
            print(f"[INFO] Beta warming: {config.beta}")

        config.logger.log({"beta": config.beta})

    def beta_cycling(self, config, epoch):
        """cycling beta btw 0 and 1 during annealing_epochs after waiting for warmup_epochs

        Arguments:
            config {namespace} -- the pipleline configuration
            epoch {integer} -- current training epoch
        """
        # TODO Callback with number of epochs
        if epoch % config.beta_annealing_epochs == 0:
            config.beta = 0
            print(f"[INFO] Beta reset to: {config.beta}")
        elif epoch % config.beta_annealing_epochs < config.beta_annealing_epochs/2:
            config.beta += 0.01/config.beta_annealing_epochs*0.5
            print(f"[INFO] Beta increased to: {config.beta}")
        else:
            print(f"[INFO] Beta constant: {config.beta}")

        config.logger.log({"beta": config.beta})
