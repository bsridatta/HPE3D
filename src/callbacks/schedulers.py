from src.callbacks import Callback


class WeightScheduler(Callback):

    def __init__(self, config, strategy):
        """Schedules weight values based on the given strategy

        Args:
            config (namespace): params for the whole pipeline, contains the beta value to schedule 
            strategy (str): Choose between cyclic and annealing. Defaults to 'cyclic'.
        """
        if "beta" in strategy:
            config.beta = 0

        if 'critic' in strategy:
            self.max_critic_weight = config.critic_weight
            self.annealing_ratio = 0.5
            config.critic_weight = 0

        if "noise" in strategy:
            self.cool_down = 20
            self.epochs_per_step = 5
            self.noise_level_max = config.noise_level

        self.strategy = strategy

    def on_train_end(self, config, epoch, **kwargs):
        getattr(WeightScheduler, f'{self.strategy}')(self, config, epoch)

    def noise_annealing(self, config, epoch):
        if (epoch-1) % self.epochs_per_step == 0:
            config.noise_level -= self.epochs_per_step*self.noise_level_max/(config.epochs-self.cool_down)  # 0.01
            print(f"[INFO] Noise Level decreased to: {config.noise_level}")

        config.logger.log({"noise": config.noise_level}, commit=False)

    def beta_annealing(self, config, epoch):
        """anneal beta from 0 to 1 during annealing_epochs after waiting for warmup_epochs

        Arguments:
            config {namespace} -- the pipleline configuration
            epoch {integer} -- current training epoch
        """
        if epoch > config.beta_warmup_epochs:
            if epoch <= config.beta_warmup_epochs + config.beta_annealing_epochs:
                config.beta += config.lambda_kld/config.beta_annealing_epochs  # 0.01
                print(f"[INFO] Beta increased to: {config.beta}")
            else:
                print(f"[INFO] Beta constant at: {config.beta}")
        else:
            print(f"[INFO] Beta warming: {config.beta}")

        config.logger.log({"beta": config.beta}, commit=False)

    def beta_cycling(self, config, epoch):
        """cycling beta btw 0 and 1 during annealing_epochs after waiting for warmup_epochs

        Arguments:
            config {namespace} -- the pipleline configuration
            epoch {integer} -- current training epoch
        """
        if epoch % config.beta_annealing_epochs == 0:
            config.beta = 0
            print(f"[INFO] Beta reset to: {config.beta}")
        elif epoch % config.beta_annealing_epochs < config.beta_annealing_epochs/2:
            config.beta += config.lambda_kld/config.beta_annealing_epochs*0.5
            print(f"[INFO] Beta increased to: {config.beta}")
        else:
            print(f"[INFO] Beta constant: {config.beta}")

        config.logger.log({"beta": config.beta}, commit=False)

    def critic_cycling(self, config, epoch):
        if epoch % config.critic_annealing_epochs == 0:
            config.critic_weight = 0
            print(f"[INFO] critic weight reset to: {config.critic_weight}")
        elif epoch % config.critic_annealing_epochs < config.critic_annealing_epochs*self.annealing_ratio:
            config.critic_weight += self.max_critic_weight/config.critic_annealing_epochs*self.annealing_ratio
            print(f"[INFO] critic weight increased to: {config.critic_weight}")
        else:
            print(f"[INFO] critic weight constant: {config.critic_weight}")

        config.logger.log({"critic_weight": config.critic_weight}, commit=False)
