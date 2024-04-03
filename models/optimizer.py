import torch
import torch.optim as optim

class OptimizerFactory:
    @staticmethod
    def build_instance(config, model) -> optim.Optimizer:
        # Load optimizer and loss - for MSE
        # optimizer = optim.Adam(network.parameters(), lr=float(config["lr"]))

        # optimizer = optim.AdamW(network.parameters(), 
        #                         lr = float(config["lr"]),
        #                         betas = (0.9, 0.999), 
        #                         eps = 1e-08, 
        #                         weight_decay = 0.01)
        # print(f'optimizer is AdamW, learning rate = {float(config["lr"])}')

        # # optimizer = optim.SGD(network.parameters(),
        # #                       lr = float(config["lr"]),
        # #                       momentum = 0.0,
        # #                       nesterov = False)
        # # print(f'optimizer is SGD, learning rate = {float(config["lr"])}')

        # AdamW
        if str(config["optimizer_type"]).lower() == 'adamw':
            lr = float(config["lr"])
            betas = (config["optimizer_betas"][0], config["optimizer_betas"][1])
            eps = float(config["optimizer_eps"])
            weight_decay = config["optimizer_weight_decay"]
            optimizer = optim.AdamW(model.parameters(), 
                                    lr = lr,
                                    betas = betas, 
                                    eps = eps, 
                                    weight_decay = weight_decay)
            print(f'Optimizer: AdamW, learning rate = {float(config["lr"])}')

        return optimizer

if __name__ == "__main__":
    print("--- Example usage ---")
    