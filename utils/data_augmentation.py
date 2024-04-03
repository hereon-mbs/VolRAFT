import numpy as np

class DVCDataAugmentation:
    # Initializer
    def __init__(self, 
                 npdtype=np.float32):
        # Super initializer
        super().__init__()

        self.npdtype = npdtype

        return
    
    @staticmethod
    def signal_power(signal):
        return np.mean( np.abs(signal) ** 2 )
    
    @staticmethod
    def add_awgn(signal, snr_db = None, noise_power = -1.0):
        '''
        Add Additive White Gaussian Noise (AWGN)
        '''
        shape = signal.shape

        # Check if the desired noise power is defined
        if noise_power <= 0.0:
            # Compute the noise_power
            power_signal = DVCDataAugmentation.signal_power(signal)

            noise_power = power_signal / (10.0 ** (snr_db / 10.0))

            print(f'Add AWGN by noise_power = {noise_power}')

        # Compute the noise
        noise = np.random.normal(loc = 0.0, 
                                scale = 1.0, 
                                size = shape) * np.sqrt(noise_power)

        # (Optional) measure the power of noise
        # measured_power_noise = signal_power(noise)

        # return the noisy signal
        return (signal + noise), noise
    
    @staticmethod
    def add_shot_noise(signal, mask = None, intensity=1.0, overshoot = 0.2):
        """
        Adds shot (Poisson) noise to an image without upper or lower limits.

        Parameters:
        signal (numpy.ndarray): The input signal array.
        mask (numpy.ndarray): A binary mask defining where to add noise (1 = add noise, 0 = no noise).
        intensity (float): A factor to control the intensity of the noise.
        overshoot (float): A factor to control the overshoot of the shot noise over the maximum of signal

        Returns:
        numpy.ndarray: The noisy signal as a floating-point array.
        """
        # Normalize the signal to have values between 0 and 1 for Poisson distribution
        if mask is None:
            mask = True
            signal_max = np.max(signal)
            signal_min = np.min(signal)
        else:
            mask = mask.astype(bool)
            signal_max = np.max(signal[mask])
            signal_min = np.min(signal[mask])

        norm_signal = np.maximum(0.0, (signal - signal_min) / (signal_max - signal_min))

        # Generate Poisson noise
        noise = np.zeros_like(signal)
        noise[mask] = np.random.poisson(norm_signal[mask] * intensity) / intensity

        # Clip the Poisson noise
        noise = np.minimum(noise, signal_max + (signal_max - signal_min) * overshoot)

        # Add the noise to the original signal
        return signal + noise * mask, noise