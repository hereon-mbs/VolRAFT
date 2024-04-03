import torch
import torch.optim as optim

class SchedulerFactory:
    @staticmethod
    def build_instance(config, optimizer) -> optim.lr_scheduler.LRScheduler:
        # StepLR
        if str(config["scheduler_type"]).lower() == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                step_size = config["scheduler_step_size"], 
                                                gamma = config["scheduler_gamma"])
            print(f'Scheduler: StepLR, step_size = {config["scheduler_step_size"]}, gamma = {config["scheduler_gamma"]}')

        # CyclicLR
        if str(config["scheduler_type"]).lower() == 'cycliclr':
            max_lr = float(config["scheduler_max_lr_factor"]) * float(config["lr"])
            step_size_up = config["scheduler_step_size_up"]  # Number of iterations to increase the learning rate
            step_size_down = config["scheduler_step_size_down"]  # Number of iterations to decrease the learning rate
            mode = config["scheduler_mode"]  # 'triangular' or 'triangular2'
            gamma = config["scheduler_gamma"]  # Multiplicative factor for learning rate decay

            scheduler = optim.lr_scheduler.CyclicLR(optimizer, 
                                                    base_lr = float(config["lr"]), 
                                                    max_lr = max_lr, 
                                                    step_size_up = step_size_up, 
                                                    step_size_down = step_size_down, 
                                                    mode = mode, 
                                                    gamma = gamma, 
                                                    cycle_momentum = False)
            print(f'Scheduler: CyclicLR, mode = {mode}, step_size = [{step_size_up}, {step_size_down}], max_lr = {max_lr}, gamma = {gamma}')

        # OneCycleLR
        if str(config["scheduler_type"]).lower() == "onecyclelr":
            max_lr = float(config["lr"])
            total_steps = int(config["scheduler_total_steps"])
            pct_start = float(config["scheduler_pct_start"])
            cycle_momentum = bool(config["scheduler_cycle_momentum"])
            anneal_strategy = str(config["scheduler_anneal_strategy"])
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr = max_lr, 
                                                    total_steps = total_steps, 
                                                    pct_start = pct_start, 
                                                    cycle_momentum = cycle_momentum, 
                                                    anneal_strategy = anneal_strategy)
            print(f'Scheduler: OneCycleLR, total_steps = {total_steps}, pct_start = {pct_start}, max_lr = {max_lr}, cycle_momentum = {cycle_momentum}, anneal_strategy = {anneal_strategy}')

        return scheduler

if __name__ == "__main__":
    print("--- Example usage ---")
    