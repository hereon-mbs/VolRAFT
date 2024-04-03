import torch
# import os

from utils.file_handler import FileHandler

## TODO: update this file
## Use this for now

class CheckpointController:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    
        return

    def save_checkpoint(self, patch_shape, flow_shape, network, optimizer, epoch, loss):
        # Define the extension
        extension = 'pt'

        # Make the file path
        checkpoint_path = FileHandler.join(self.checkpoint_dir, 
                                    f'checkpoint_epoch{epoch:06d}.{extension}')
        
        # Save the checkpoint
        # TODO: include scheduler
        torch.save({
            'epoch': epoch,
            'patch_shape': patch_shape,
            'flow_shape': flow_shape,
            'network_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path)
        return

    def load_checkpoint(self, filename, network, optimizer, scheduler):
        path = FileHandler.join(self.checkpoint_dir, filename)
        print(f'load checkpoint file: {path}')

        # Load checkpoint
        checkpoint = torch.load(path)

        # Load state dictionary
        epoch = checkpoint['epoch']
        patch_shape = checkpoint['patch_shape']
        flow_shape = checkpoint['flow_shape']

        if network is not None:
            network.load_state_dict(checkpoint['network_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        loss_list_train = checkpoint['loss_list_train']
        loss_list_valid = checkpoint['loss_list_valid']

        return epoch, patch_shape, flow_shape, network, optimizer, scheduler, loss_list_train, loss_list_valid

    def load_last_checkpoint(self, network, optimizer, scheduler):
        checkpoint_files = FileHandler.listdir(FileHandler.realpath(self.checkpoint_dir))
        if not checkpoint_files:
            print("No checkpoint files found in the directory.")
            return network, optimizer, scheduler
        
        last_checkpoint = ""

        for file in checkpoint_files:
            if file.endswith(".pt"):
                if file > last_checkpoint:
                    last_checkpoint = file

        return self.load_checkpoint(last_checkpoint, network, optimizer, scheduler)
    
    def find_config_file(self) -> str:
        checkpoint_files = FileHandler.listdir(self.checkpoint_dir)

        if not checkpoint_files:
            print("No checkpoint files found in the directory.")
            return None
        
        config_file = ""

        for file in checkpoint_files:
            if file.endswith(".yaml"):
                if file > config_file:
                    config_file = file

        return FileHandler.join(self.checkpoint_dir, config_file)

if __name__ == "__main__":
    print("--- Example usage ---")
    # Dummy model and optimizer for demonstration
    network = torch.nn.Linear(10, 5)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)

    # Create an instance of CheckpointController
    checkpoint_manager = CheckpointController()

    # Save a checkpoint (example: epoch 5)
    checkpoint_manager.save_checkpoint(patch_shape = (1, 1, 1),
                                        flow_shape = (1, 1, 1),
                                        network = network,
                                        optimizer = optimizer, 
                                        epoch = 5,
                                        loss = list(range(5)))
    
    checkpoint_manager.save_checkpoint(patch_shape = (1, 1, 1),
                                    flow_shape = (1, 1, 1),
                                    network = network,
                                    optimizer = optimizer, 
                                    epoch = 10,
                                    loss = list(range(10)))

    # Load a checkpoint (example: loading epoch 5)
    loaded_network, loaded_optimizer, loaded_epoch, loaded_loss = checkpoint_manager.load_checkpoint('checkpoint_epoch000005.pt', network, optimizer)

    print(f'loaded_network: {loaded_network}')
    print(f'loaded_optimizer: {loaded_optimizer}')
    print(f'loaded_epoch: {loaded_epoch}')
    print(f'loaded_loss: {loaded_loss}')

    print("--- Example usage ---")
    # Load a checkpoint (example: loading epoch 5)
    loaded_network, loaded_optimizer, loaded_epoch, loaded_loss = checkpoint_manager.load_last_checkpoint(network, optimizer)

    print(f'loaded_network: {loaded_network}')
    print(f'loaded_optimizer: {loaded_optimizer}')
    print(f'loaded_epoch: {loaded_epoch}')
    print(f'loaded_loss: {loaded_loss}')