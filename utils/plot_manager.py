import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import numpy as np

class PlotManager:
    def __init__(self, figsize=(32, 12), fontsize=24):
        """
        Initializes the PlotManager with any necessary configuration.
        """

        # Define a proper fontsize
        self.fontsize = fontsize

        # Define a proper figure size
        self.figsize = figsize

        # Set default styles or any other matplotlib configurations here
        # plt.style.use('seaborn-darkgrid')
    
    @staticmethod
    def plot_image(image: np.ndarray, title: str = "Image", cmap: str = 'gray') -> None:
        """
        Plots a single image.

        Parameters:
        - image (np.ndarray): The image to plot.
        - title (str): Title of the plot.
        - cmap (str): Colormap used for the plot.
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    @staticmethod
    def plot_images_grid(images: np.ndarray, ncols: int = 3, figsize: tuple = (12, 8), cmap: str = 'gray') -> None:
        """
        Plots a grid of images.

        Parameters:
        - images (np.ndarray): Array of images to plot. Assumed to be in the format [N, H, W] or [N, H, W, C].
        - ncols (int): Number of columns in the image grid.
        - figsize (tuple): Figure size.
        - cmap (str): Colormap used for the plots.
        """
        n_images = images.shape[0]
        nrows = n_images // ncols + (n_images % ncols > 0)

        plt.figure(figsize=figsize)
        for i in range(n_images):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(images[i], cmap=cmap)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    
    ####################### Old Version ##################################
    # Define function to plot multiple images
    def plot_images(self, images, 
                    titles=None, 
                    limits=[], 
                    in_log_scale=False, 
                    num_subplots=None, 
                    caxis_mode='None', 
                    each_figsize=(8, 8), 
                    hide_axis=True,
                    sharex=True, sharey=True,
                    cmap='viridis'):
        """
        Parameters:
        -----------
        images: array of numpy array
            Images to be plotted
        titles: array of string, optional
            Titles of images
        limits: array of tuple, optional
            Array of plot limits (in tuple)
            When it is only one element (i.e. only 1 tuple), 
            all plots share the same limit
        in_log_scale: boolean, optional
            Indicate whether images are transformed into logarithmic scale
        num_subplots: tuple, optional
            The number of subplots in (x, y) directions
        caxis_mode: string, optional
            Indicate whether the colorbar mode is 
            "None", "(V)ertical", or "(H)orizontal" (case insensitive)
        each_figsize: tuple, optional
            The size of subfigure
        sharex: boolean, optional
            Indicate whether subplots share the x-axis
        sharey: boolean, optional
            Indicate whether subplots share the y-axis
        cmap: string, optional
            Indicate the colormaps of all subplots
        """
        # Control colorbar
        if 'v' in caxis_mode.lower():
            # Vertical colorbar
            caxis_on = True
            caxis_orientation = 'vertical'
        elif 'h' in caxis_mode.lower():
            # Horizontal colorbar
            caxis_on = True
            caxis_orientation = 'horizontal'
        else:
            # Default no colorbar
            caxis_on = False
            caxis_orientation = 'vertical'

        # Control number of subplots
        if num_subplots is None:
            num_subplots = (1, len(images))

        # Control overall figsize
        figsize = (each_figsize[0] * num_subplots[1], 
                   each_figsize[1] * num_subplots[0])
            
        # Get number of images
        num_images = len(images)

        # Create figure
        fig, axs = plt.subplots(nrows=num_subplots[0], 
                                ncols=num_subplots[1], 
                                figsize=figsize, 
                                sharex=sharex, 
                                sharey=sharey)

        # Prepare handles for image and axe objects
        handle_img = []
        handle_axs = []
        if num_images > 1:
            for idx, ax in enumerate(axs):
                handle_axs.append(ax)
        else:
            handle_axs.append(axs)

        # Plot figures
        for idx, ax in enumerate(handle_axs):
            # Show image or show in logarithmic scale
            if in_log_scale:
                handle_img.append(
                    ax.imshow(10*np.log10(images[idx], where=images[idx] > 0), cmap=cmap)
                    )
            else:
                handle_img.append(ax.imshow(images[idx], cmap=cmap))

            # Set colorbar
            if caxis_on:
                fig.colorbar(handle_img[idx], 
                             fraction=0.046, 
                             pad=0.04, 
                             orientation=caxis_orientation)
            
            # Set title
            if titles is None:
                ax.set_title(f'image {idx}')
            else:
                ax.set_title(titles[idx])

            # Hide axis
            if hide_axis:
                ax.axis('off')

            # Set color limits
            if limits is not None:
                if len(limits) > 1:
                    # individual limits
                    handle_img[idx].set_clim(vmin=limits[idx][0], 
                                             vmax=limits[idx][1])
                elif len(limits) == 1:
                    # only one limit
                    handle_img[idx].set_clim(vmin=limits[0][0], 
                                             vmax=limits[0][1])

        # Fine tune the layout
        fig.tight_layout()

        # Show figure
        # plt.show()

        # Return objects
        return fig, axs

    def plot_images_simple(self, 
                           images, 
                           titles=None, 
                           ncols=3, 
                           figsize=(10, 10), 
                           save_to_file=False, 
                           file_name="image"):
        num_images = len(images)
        nrows = (num_images + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        for i in range(num_images):
            row = i // ncols
            col = i % ncols

            if titles is not None:
                title = titles[i]
            else:
                title = f"Image {i+1}"

            if isinstance(images[i], torch.Tensor):
                image = images[i].cpu().numpy()
            elif isinstance(images[i], np.ndarray):
                image = images[i]
            else:
                raise ValueError("Unsupported image type. Must be a PyTorch tensor or a NumPy array.")

            if image.shape[0] == 1:
                image = image[0]

            if nrows == 1:
                axes[col].imshow(image, cmap='gray') if len(image.shape) == 2 else axes[col].imshow(image)
                axes[col].set_title(title)
                axes[col].axis('off')
            else:
                axes[row, col].imshow(image, cmap='gray') if len(image.shape) == 2 else axes[row, col].imshow(image)
                axes[row, col].set_title(title)
                axes[row, col].axis('off')

        plt.tight_layout()

        if save_to_file:
            plt.savefig(f"{file_name}.png")
        else:
            plt.show()

    def plot_loss(self, epoch_indices, 
                  loss_train, loss_valid, 
                  epoch_validation = 1):
        plt.close()

        # Increase figure size
        fig = plt.figure(figsize=self.figsize)

        # Plot Training and Validation Loss on the same figure
        plt.plot(epoch_indices, loss_train, label='Training Loss', color='C0')
        plt.plot(epoch_indices[::epoch_validation], loss_valid[::epoch_validation], label='Validation Loss', color='C1')
        plt.title('Training and Validation Loss over Epochs', fontsize=self.fontsize)
        plt.xlabel('Epochs', fontsize=self.fontsize)
        plt.ylabel('Loss', fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        if epoch_indices.max() > epoch_indices.min():
            plt.xlim([epoch_indices.min(), epoch_indices.max()])
        plt.legend(fontsize=self.fontsize)

        # Adjust layout for better presentation
        plt.tight_layout()

        return
    
    def plot_metrics(self, epoch_indices, 
                     metrics_train, metrics_valid, 
                     metrics_name = ['EPE', 'Endpoint Error'],
                     epoch_validation = 1):
        plt.close()

        # Increase figure size
        fig = plt.figure(figsize=self.figsize)

        # Plot Training and Validation Loss on the same figure
        plt.plot(epoch_indices, metrics_train, label=f'Training {metrics_name[0]}', color='C0')
        plt.plot(epoch_indices[::epoch_validation], metrics_valid[::epoch_validation], label=f'Validation {metrics_name[0]}', color='C1')
        plt.title(f'Training and Validation {metrics_name[1]} over Epochs', fontsize=self.fontsize)
        plt.xlabel('Epochs', fontsize=self.fontsize)
        plt.ylabel(f'{metrics_name[1]} ({metrics_name[0]})', fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        if epoch_indices.max() > epoch_indices.min():
            plt.xlim([epoch_indices.min(), epoch_indices.max()])
        plt.legend(fontsize=self.fontsize)

        # Adjust layout for better presentation
        plt.tight_layout()

        return
    
    def plot_lr(self, epoch_indices, lr):
        plt.close()

        # Set formatter to ScalarFormatter for both axes
        # formatter = ticker.ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True)
        # formatter.set_powerlimits((-10, 10))  # You can adjust these limits as needed

        # Increase figure size
        fig = plt.figure(figsize=self.figsize)

        # Plot Learning Rate
        plt.plot(epoch_indices, lr, label='Learning Rate', color='C3')
        plt.title('Learning Rate over Epochs', fontsize=self.fontsize)
        plt.xlabel('Epochs', fontsize=self.fontsize)
        plt.ylabel('Learning Rate', fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        if epoch_indices.max() > epoch_indices.min():
            plt.xlim([epoch_indices.min(), epoch_indices.max()])
        plt.legend(fontsize=self.fontsize)

        # plt.gca().yaxis.set_major_formatter(formatter)

        # Custom formatter function to format labels as "1.2 x 10^-4"
        def scientific_formatter(y, pos):
            """ Custom formatter to display y-axis labels in scientific notation """
            if y == 0:
                return r'$0$'

            # Handle very small values that are effectively zero
            if np.abs(y) < 1e-14:
                return r'$0$'
    
            # Determine the exponent, which is the floor of the logarithm of the absolute max value
            exponent = np.floor(np.log10(np.abs(y)))
            # Make the y-value a fraction of the form 1.x, 2.x, etc.
            frac = y / 10**exponent
            return r'${:.1f} \times 10^{{{:.0f}}}$'.format(frac, exponent)

        # Set the custom formatter
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(scientific_formatter))

        # Adjust layout for better presentation
        plt.tight_layout()

        return    

# Example usage:
if __name__ == "__main__":
    plot_manager = PlotManager()
    # # Assuming 'image' is a numpy array representing an image
    # plot_manager.plot_image(image, "Sample Image")

    # # Assuming 'images' is a numpy array of shape [N, H, W] or [N, H, W, C] representing multiple images
    # plot_manager.plot_images_grid(images, ncols=4)
