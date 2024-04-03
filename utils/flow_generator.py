# import torch
import numpy as np
# import os

from utils.device_manager import DeviceManager
from utils.mask import DVCMask
from utils.flow import DVCFlow

class DVCFlowGenerator:
    # Initializer
    def __init__(self, 
                 npdtype=np.float32):
        # Super initializer
        super().__init__()

        self.npdtype = npdtype

        return
    
    @staticmethod
    def find_mask_box(data_np):
        # Find the coordinates where the mask is True
        coordinates = np.argwhere(data_np)

        # Center of mask
        center = np.mean(coordinates, axis = 0)
        # center = np.median(coordinates, axis = 0) # Use median instead

        # Bounding box
        box_min = np.min(coordinates, axis = 0).astype(np.float32)
        box_max = np.max(coordinates, axis = 0).astype(np.float32)

        # Normalize to the size of data
        shape = np.array(data_np.shape).astype(np.float32)
        center_norm = center / shape
        box_min_norm = box_min / shape
        box_max_norm = box_max / shape

        print(f'mask box center (x, y, z) = {center_norm}')
        print(f'mask box min (x, y, z) = {box_min_norm}')
        print(f'mask box max (x, y, z) = {box_max_norm}')

        return center_norm, box_min_norm, box_max_norm
    
    # Additive White Gaussian Noise
    @staticmethod
    def add_awgn(signal, 
                snr_db = None, 
                noise_power = -1.0):
        def signal_power(signal):
            return np.mean( np.abs(signal) ** 2 )
        
        shape = signal.shape

        # Check if the desired noise power is defined
        if noise_power <= 0.0:
            # Compute the noise_power
            power_signal = signal_power(signal)

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
    def generate_random_gaussian(center, min, max):
        # Calculate the standard deviation so that 3-sigma covers the range from min to max
        sigma = (max - min) / 6

        # Generate a random number with Gaussian distribution
        return np.random.normal(center, sigma)

    @staticmethod
    def generate_synthetic(shape, indexing, npdtype, flow_config, mask=None) -> DVCFlow:
        print(f'Generate synthetic flow field for {shape}')

        data_np = None

        # Max flow is always needed for any synthetic field
        # Default value: 1% of volumetric dimension (Euclidean distance)
        # Default value: a random number in the shortest dimension
        if "max" not in flow_config:
            # flow_config["max"] = np.sqrt(np.sum(np.array(shape) ** 2)) * 0.01
            # shape_dim = np.array(shape).min() / 16.0
            # flow_config["max"] = DVCFlowGenerator.generate_random_gaussian(center = shape_dim / 2, min = 1, max = shape_dim)

            flow_config["max"] = DVCFlowGenerator.generate_random_gaussian(center = 18.0, min = 12.0, max = 24.0)

        # Accept string or integer
        if flow_config["type"].lower() == 'star' or flow_config["type"] == 0:
            data_np = DVCFlowGenerator.synthetic_star(shape, flow_config = flow_config)

        # Accept string or integer
        if flow_config["type"].lower() == 'static' or flow_config["type"] == 1:
            if "displacement" not in flow_config:
                flow_config["displacement"] = (np.random.random(size=(3)) * 2.0 - 1.0) * flow_config["max"]
                print(f'displacement = {flow_config["displacement"]}')

            data_np = DVCFlowGenerator.synthetic_static(shape, flow_config)

        # Accept string or integer
        if flow_config["type"].lower() == 'curve' or flow_config["type"] == 2:
            if "slope_ratio" not in flow_config:
                flow_config["slope_ratio"] = 0.8
                print(f'slope_ratio = {flow_config["slope_ratio"]}')

            data_np = DVCFlowGenerator.synthetic_curve(shape, flow_config)

        # Accept string or integer
        if flow_config["type"].lower() == 'sphere' or flow_config["type"] == 3:
            data_np = DVCFlowGenerator.synthetic_sphere(shape, flow_config, mask=mask)

        # Accept string or integer
        if flow_config["type"].lower() == 'overall' or flow_config["type"] == 4:
            data_np = DVCFlowGenerator.synthetic_overall(shape, flow_config, mask=mask)
            
        # Save the default indexing
        flow = DVCFlow(flow_data = data_np,
                       indexing = indexing,
                       npdtype = npdtype)
        
        DeviceManager.garbage()

        return flow
    
    # Type of flow fields
    @staticmethod
    def synthetic_star(shape, flow_config, npdtype = np.float32) -> np.ndarray:
        X, Y, Z = shape

        print(f'star flow field {X} x {Y} x {Z} for max_flow = {flow_config["max"]}')

        # Normalized grid [0, 1)
        gx = np.arange(X) / X
        gy = np.arange(Y) / Y
        gz = np.arange(Z) / Z
        
        # Normalized mesh [0, 1)
        mesh_x, mesh_y, mesh_z = np.meshgrid(gx, gy, gz, sparse=True, indexing='ij')

        # Amplitude of flow (in terms of voxels)
        a1 = flow_config["max"] - np.random.random() * 0.1
        a2 = flow_config["max"] - np.random.random() * 0.1
        a3 = flow_config["max"] - np.random.random() * 0.1
        print(f'star field (a1, a2, a3) = {a1}, {a2}, {a3}')

        # Midpoint
        x0 = 0.5 + (np.random.random() * 2.0 - 1.0) * 0.1
        y0 = 0.5 + (np.random.random() * 2.0 - 1.0) * 0.1
        z0 = 0.5 + (np.random.random() * 2.0 - 1.0) * 0.1
        print(f'star field (x0, y0, z0) = {x0}, {y0}, {z0}')

        # Period
        T_min = 0.0625 + (np.random.rand() * 0.01)
        T_max = 0.375  + (np.random.rand() * 0.1)
        print(f'star field (T) = [{T_min}, {T_max}]')

        Tx = T_min + ((T_max - T_min) * mesh_y)
        Ty = T_min + ((T_max - T_min) * mesh_z)
        Tz = T_min + ((T_max - T_min) * mesh_x)

        # Get a list of gradients
        d = []
        d.append(np.array(a1 * np.sin( 2 * np.pi * (mesh_z - x0) / Tx ) + mesh_x * 0.0 + mesh_y * 0.0, dtype=npdtype))
        d.append(np.array(a2 * np.sin( 2 * np.pi * (mesh_x - y0) / Ty ) + mesh_y * 0.0 + mesh_z * 0.0, dtype=npdtype))
        d.append(np.array(a3 * np.cos( 2 * np.pi * (mesh_y - z0) / Tz ) + mesh_x * 0.0 + mesh_z * 0.0, dtype=npdtype))

        # Shuffle dx, dy, dz
        if "orders" in flow_config:
            orders = flow_config["orders"]
        else:
            orders = np.random.permutation([0, 1, 2])

        print(f'start field orders = {orders}')

        # Concatenate
        data_np = np.concatenate((
            np.expand_dims(d[orders[0]], 0),
            np.expand_dims(d[orders[1]], 0),
            np.expand_dims(d[orders[2]], 0),
        ), axis = 0).astype(npdtype)

        return data_np
    
    # Type of flow fields
    @staticmethod
    def synthetic_static( shape, flow_config, npdtype = np.float32) -> np.ndarray:
        X, Y, Z = shape

        print(f'static flow field {X} x {Y} x {Z} for max_flow = {flow_config["max"]}')

        # Prepare array
        dx = np.full(shape = (X, Y, Z), fill_value = flow_config["displacement"][0], dtype = npdtype)
        dy = np.full(shape = (X, Y, Z), fill_value = flow_config["displacement"][1], dtype = npdtype)
        dz = np.full(shape = (X, Y, Z), fill_value = flow_config["displacement"][2], dtype = npdtype)

        print(f'static transform flow field {X} x {Y} x {Z} for {flow_config["displacement"]}')

        # Add noise (optional)
        if "snr_db" in flow_config:
            dx, _ = DVCFlowGenerator.add_awgn(dx, snr_db = flow_config["snr_db"])
            dy, _ = DVCFlowGenerator.add_awgn(dy, snr_db = flow_config["snr_db"])
            dz, _ = DVCFlowGenerator.add_awgn(dz, snr_db = flow_config["snr_db"])

            print(f'static transform flow field dx range = [{np.min(dx)}, {np.max(dx)}]')
            print(f'static transform flow field dy range = [{np.min(dy)}, {np.max(dy)}]')
            print(f'static transform flow field dz range = [{np.min(dz)}, {np.max(dz)}]')

        # Get a list of gradients
        d = []
        d.append(dx)
        d.append(dy)
        d.append(dz)

        # Shuffle dx, dy, dz
        if "orders" in flow_config:
            orders = flow_config["orders"]
        else:
            orders = np.random.permutation([0, 1, 2])

        print(f'static field orders = {orders}')
            
        # Concatenate
        data_np = np.concatenate((
            np.expand_dims(d[orders[0]], 0),
            np.expand_dims(d[orders[1]], 0),
            np.expand_dims(d[orders[2]], 0),
        ), axis = 0).astype(npdtype)

        return data_np
    
    # Type of flow fields
    @staticmethod
    def synthetic_curve(shape, flow_config, npdtype = np.float32) -> np.ndarray:
        X, Y, Z = shape

        print(f'curve flow field {X} x {Y} x {Z} for max_flow = {flow_config["max"]}')        

        # Normalized grid [0, 1)
        gx = np.arange(X) / X
        gy = np.arange(Y) / Y
        gz = np.arange(Z) / Z
        
        # Normalized mesh [0, 1)
        # mesh_x, mesh_y, mesh_z = np.meshgrid(gx, gy, gz, indexing='ij')
        mesh_x, mesh_y, mesh_z = np.meshgrid(gx, gy, gz, sparse=True, indexing='ij')
        # mesh_x, mesh_y, mesh_z = np.mgrid[0:1:1/X, 0:1:1/Y, 0:1:1/Z]

        # Prepare line f(x) = m * x + c
        
        if "gamma" in flow_config:
            gamma = flow_config["gamma"]
        else:
            gamma = 1.0 + (np.random.random(size=(3)) * 1.0)

        # Slopes
        if "slope" in flow_config:
            slope = flow_config["slope"]
        else:
            max_slope = flow_config["max"] * np.minimum(1.0, flow_config["slope_ratio"])
            slope = (np.random.random(size=(3)) * 2.0 - 1.0) * max_slope

        # Shifts
        if "shift" in flow_config:
            shift = flow_config["shift"]
        else:
            max_shift = flow_config["max"] * (1.0 - np.minimum(1.0, flow_config["slope_ratio"]))
            shift = (np.random.random(size=(3)) * 2.0 - 1.0) * max_shift

        # Get a list of gradients
        d = []
        # add redundant meshes together to broadcast the matrices
        d.append(np.array(slope[0] * (mesh_x ** gamma[0]) + shift[0] + mesh_y * 0.0 + mesh_z * 0.0, dtype=npdtype))
        d.append(np.array(slope[1] * (mesh_y ** gamma[1]) + shift[1] + mesh_x * 0.0 + mesh_z * 0.0, dtype=npdtype))
        d.append(np.array(slope[2] * (mesh_z ** gamma[2]) + shift[2] + mesh_x * 0.0 + mesh_y * 0.0, dtype=npdtype))

        print(f'curve field dx = {slope[0]:.4f} * x^{gamma[0]:.4f} + {shift[0]:.4f}, range = [{np.min(d[0]):.4f}, {np.max(d[0]):.4f}]')
        print(f'curve field dy = {slope[1]:.4f} * y^{gamma[1]:.4f} + {shift[1]:.4f}, range = [{np.min(d[1]):.4f}, {np.max(d[1]):.4f}]')
        print(f'curve field dz = {slope[2]:.4f} * z^{gamma[2]:.4f} + {shift[2]:.4f}, range = [{np.min(d[2]):.4f}, {np.max(d[2]):.4f}]')

        # Shuffle dx, dy, dz
        if "orders" in flow_config:
            orders = flow_config["orders"]
        else:
            orders = np.random.permutation([0, 1, 2])

        print(f'curve field orders = {orders}')

        # Concatenate
        data_np = np.concatenate((
            np.expand_dims(d[orders[0]], 0),
            np.expand_dims(d[orders[1]], 0),
            np.expand_dims(d[orders[2]], 0),
        ), axis = 0).astype(npdtype)

        # Rotate the flow field
        if "rotation" in flow_config:
            rotation = flow_config["rotation"]
        else:
            rotation = np.random.random(size=(3)) * 360.0

        print(f'curve field rotated by degrees: {rotation[0]:.4f}, {rotation[1]:.4f}, {rotation[2]:.4f}')

        data_np = DVCFlow.rotate_flow(data_np, 
                                      rotation = np.radians(rotation))

        return data_np.astype(npdtype)
    
    @staticmethod
    def synthetic_sphere(shape, flow_config, mask=None, npdtype = np.float32) -> np.ndarray:
        
        X, Y, Z = shape

        print(f'sphere flow field {X} x {Y} x {Z} for max_flow = {flow_config["max"]}')

        # Normalized grid [0, 1)
        gx = np.arange(X) / X
        gy = np.arange(Y) / Y
        gz = np.arange(Z) / Z
        
        # Normalized mesh [0, 1)
        mesh_x, mesh_y, mesh_z = np.meshgrid(gx, gy, gz, sparse=True, indexing='ij')

        # Scale
        if "scale" in flow_config:
            scale = flow_config["scale"]
        else:
            scale = 1.0 + (np.random.random(size=(3)) * 1.0)

        # Center
        if "center" in flow_config:
            center = flow_config["center"]
        else:
            if mask is None:
                center = np.random.random(size=(3))
            else:
                mask_box_center, mask_box_min, mask_box_max = DVCFlowGenerator.find_mask_box(mask.data_np.squeeze())
                # center = np.random.random(size=(3)) * (mask_box_max - mask_box_min) + mask_box_min
                center = DVCFlowGenerator.generate_random_gaussian(center = mask_box_center, 
                                                                   min = mask_box_min, 
                                                                   max = mask_box_max)

        # Velocity
        if "velocity" in flow_config:
            velocity = flow_config["velocity"]
        else:
            velocity = np.random.random(size=(3)) * 2.0 - 1.0

        # Phase
        if "phase_pi" in flow_config:
            phase = flow_config["phase_pi"] * np.pi
        else:
            phase = np.random.random() * 2.0 * np.pi

        # Shuffle dx, dy, dz
        if "orders" in flow_config:
            orders = flow_config["orders"]
        else:
            orders = np.random.permutation([0, 1, 2])

        print(f'sphere flow field center at = {center}')
        print(f'sphere flow field scale = {scale}')
        print(f'sphere flow field velocity = {velocity}')
        print(f'sphere flow field phase = {phase / np.pi} * pi')

        # Shift and scale the coordinates
        x_scaled = (mesh_x - center[0]) / scale[0]
        y_scaled = (mesh_y - center[1]) / scale[1]
        z_scaled = (mesh_z - center[2]) / scale[2]

        # Convert scaled Cartesian coordinates to spherical coordinates
        r = np.sqrt(x_scaled**2 + y_scaled**2 + z_scaled**2)

        # Ensure no division by zero
        r[r == 0] = np.finfo(float).eps

        theta = np.arccos(z_scaled / r)
        phi = np.arctan2(y_scaled, x_scaled)

        # Radial component (set to 0 for purely tangential flow)
        vrx = np.ones_like(r) * velocity[0]
        vry = np.ones_like(r) * velocity[1]
        vrz = np.ones_like(r) * velocity[2]

        # Tangential components (circular patterns around the sphere)
        vtheta = -np.sin(phi + phase)  # Circular pattern in polar direction
        vphi = np.cos(theta + phase)  # Circular pattern in azimuthal direction

        # Convert the spherical components back to Cartesian coordinates for the vector field
        u = (vrx * np.sin(theta) * np.cos(phi) + \
            vtheta * np.cos(theta) * np.cos(phi) - \
            vphi * np.sin(phi)) * scale[0]

        v = (vry * np.sin(theta) * np.sin(phi) + \
            vtheta * np.cos(theta) * np.sin(phi) + \
            vphi * np.cos(phi)) * scale[1]

        w = (vrz * np.cos(theta) - \
            vtheta * np.sin(theta)) * scale[2]
        
        mesh_x = mesh_y = mesh_z = None
        x_scaled = y_scaled = z_scaled = None
        r = None
        theta = phi = None
        vrx = vry = vrz = None
        DeviceManager.garbage()
        
        uvw_max = np.max([np.abs(u).max(), np.abs(v).max(), np.abs(w).max()])

        print(f'sphere flow field max = {uvw_max}')

        # Get a list of gradients
        d = []
        # add redundant meshes together to broadcast the matrices
        d.append(u / uvw_max * flow_config["max"])
        u = None
        DeviceManager.garbage()
        
        d.append(v / uvw_max * flow_config["max"])
        v = None
        DeviceManager.garbage()

        d.append(w / uvw_max * flow_config["max"])
        w = None
        DeviceManager.garbage()

        print(f'sphere field orders = {orders}')

        # Concatenate
        data_np = np.concatenate((
            np.expand_dims(d[orders[0]], 0),
            np.expand_dims(d[orders[1]], 0),
            np.expand_dims(d[orders[2]], 0),
        ), axis = 0).astype(npdtype)

        DeviceManager.garbage()

        return data_np
    
    # Type of flow fields
    @staticmethod
    def synthetic_overall(shape, flow_config, mask=None, npdtype = np.float32) -> np.ndarray:
        X, Y, Z = shape

        print(f'overall flow field {X} x {Y} x {Z} for max_flow = {flow_config["max"]}')

        # Normalized grid [0, 1)
        gx = np.arange(X, dtype = npdtype) / X
        gy = np.arange(Y, dtype = npdtype) / Y
        gz = np.arange(Z, dtype = npdtype) / Z
        
        # Normalized mesh [0, 1)
        mesh_x, mesh_y, mesh_z = np.meshgrid(gx, gy, gz, sparse=True, indexing='ij')

        # Scale
        if "scale" in flow_config:
            scale = flow_config["scale"]
        else:
            scale = 1.0 + (np.random.random(size=(3)) * 1.0)

        # Center
        if "center" in flow_config:
            center = flow_config["center"]
        else:
            if mask is None:
                center = np.random.random(size=(3))
            else:
                mask_box_center, mask_box_min, mask_box_max = DVCFlowGenerator.find_mask_box(mask.data_np.squeeze())
                # center = np.random.random(size=(3)) * (mask_box_max - mask_box_min) + mask_box_min
                center = DVCFlowGenerator.generate_random_gaussian(center = mask_box_center, 
                                                                   min = mask_box_min, 
                                                                   max = mask_box_max)

        # Velocity
        if "velocity" in flow_config:
            velocity = flow_config["velocity"]
        else:
            velocity = np.random.random(size=(3)) * 2.0 - 1.0

        # Phase
        if "phase_pi" in flow_config:
            phase = flow_config["phase_pi"] * np.pi
        else:
            phase = np.random.random() * 2.0 * np.pi

        # Gamma
        if "gamma" in flow_config:
            gamma = flow_config["gamma"]
        else:
            gamma = 1.0 + (np.random.random(size=(3)) * 1.0)

        # Shuffle dx, dy, dz
        if "orders" in flow_config:
            orders = flow_config["orders"]
        else:
            orders = np.random.permutation([0, 1, 2])

        # Rotation
        if "rotation" in flow_config:
            rotation = flow_config["rotation"]
        else:
            rotation = np.random.random(size=(3)) * 360.0

        # Star field weight
        if "star_weight" in flow_config:
            star_weight = flow_config["star_weight"]
        else:
            # star_weight = np.random.random() * 0.5
            star_weight = DVCFlowGenerator.generate_random_gaussian(center = 0.1, min = 0.0, max = 0.2)

        # Shuffle dx, dy, dz for the star_field
        if "star_orders" in flow_config:
            star_orders = flow_config["star_orders"]
        else:
            star_orders = np.random.permutation([0, 1, 2])

        # Rotation for star field
        if "star_rotation" in flow_config:
            star_rotation = flow_config["star_rotation"]
        else:
            star_rotation = np.random.random(size=(3)) * 360.0

        print(f'overall flow field center at = {center}')

        # Shift and scale the coordinates
        x_scaled = ((mesh_x - center[0]) / (scale[0])).astype(npdtype)
        y_scaled = ((mesh_y - center[1]) / (scale[1])).astype(npdtype)
        z_scaled = ((mesh_z - center[2]) / (scale[2])).astype(npdtype)

        # Convert scaled Cartesian coordinates to spherical coordinates
        r = np.sqrt(x_scaled**2 + y_scaled**2 + z_scaled**2)

        # Ensure no division by zero
        r[r == 0] = np.finfo(float).eps

        print(f'overall flow field scale = {scale}')

        theta = np.arccos(z_scaled / r)
        phi = np.arctan2(y_scaled, x_scaled)

        # Radial component (set to 0 for purely tangential flow)
        vrx = np.ones_like(r, dtype=npdtype) * velocity[0]
        vry = np.ones_like(r, dtype=npdtype) * velocity[1]
        vrz = np.ones_like(r, dtype=npdtype) * velocity[2]

        print(f'overall flow field velocity = {velocity}')

        # Tangential components (circular patterns around the sphere)
        vtheta = -np.sin(phi + phase)  # Circular pattern in polar direction
        vphi = np.cos(theta + phase)  # Circular pattern in azimuthal direction

        print(f'overall flow field phase = {phase / np.pi} * pi')

        # Convert the spherical components back to Cartesian coordinates for the vector field
        u = (vrx * np.sin(theta) * np.cos(phi) + \
            vtheta * np.cos(theta) * np.cos(phi) - \
            vphi * np.sin(phi)) * scale[0] * ((1.0 + mesh_x) ** gamma[0] - mesh_x)

        v = (vry * np.sin(theta) * np.sin(phi) + \
            vtheta * np.cos(theta) * np.sin(phi) + \
            vphi * np.cos(phi)) * scale[1] * ((1.0 + mesh_y) ** gamma[1] - mesh_y)

        w = (vrz * np.cos(theta) - \
            vtheta * np.sin(theta)) * scale[2] * ((1.0 + mesh_z) ** gamma[2] - mesh_z)
        
        mesh_x = mesh_y = mesh_z = None
        x_scaled = y_scaled = z_scaled = None
        r = None
        theta = phi = None
        vrx = vry = vrz = None
        DeviceManager.garbage()

        print(f'overall flow field gamma = {gamma}')
        
        # Generate a star field
        star_flow_config = {
            "type": "star",
            "max": flow_config["max"],
            "orders": star_orders
            }
        star_data_np = DVCFlowGenerator.synthetic_star(shape, star_flow_config).astype(npdtype)

        print(f'star field rotated by degrees: {star_rotation[0]:.4f}, {star_rotation[1]:.4f}, {star_rotation[2]:.4f}')

        star_data_np = DVCFlow.rotate_flow(star_data_np, rotation = np.radians(star_rotation)).astype(npdtype)

        u += star_weight * ((star_data_np[0, :, :, :] / flow_config["max"]) * 2.0 - 1.0)
        v += star_weight * ((star_data_np[1, :, :, :] / flow_config["max"]) * 2.0 - 1.0)
        w += star_weight * ((star_data_np[2, :, :, :] / flow_config["max"]) * 2.0 - 1.0)

        print(f'star field weight = {star_weight}')

        DeviceManager.garbage()
        
        # Add noise (optional)
        if "snr_db" in flow_config:
            u, _ = DVCFlowGenerator.add_awgn(u, snr_db = flow_config["snr_db"])
            DeviceManager.garbage()
            v, _ = DVCFlowGenerator.add_awgn(v, snr_db = flow_config["snr_db"])
            DeviceManager.garbage()
            w, _ = DVCFlowGenerator.add_awgn(w, snr_db = flow_config["snr_db"])
            DeviceManager.garbage()
            print(f'overall flow field AWGN SNR = {flow_config["snr_db"]} dB')
        
        uvw_max = np.max([np.abs(u).max(), np.abs(v).max(), np.abs(w).max()])

        print(f'overall flow field max = {uvw_max}')

        # Get a list of gradients
        d = []
        # add redundant meshes together to broadcast the matrices
        d.append(np.array(u / uvw_max * flow_config["max"], dtype=npdtype))
        u = None
        DeviceManager.garbage()

        d.append(np.array(v / uvw_max * flow_config["max"], dtype=npdtype))
        v = None
        DeviceManager.garbage()

        d.append(np.array(w / uvw_max * flow_config["max"], dtype=npdtype))
        w = None
        DeviceManager.garbage()

        print(f'overall field orders = {orders}')

        # Concatenate
        data_np = np.concatenate((
            np.expand_dims(d[orders[0]], 0),
            np.expand_dims(d[orders[1]], 0),
            np.expand_dims(d[orders[2]], 0),
        ), axis = 0).astype(npdtype)

        print(f'overall field rotated by degrees: {rotation[0]:.4f}, {rotation[1]:.4f}, {rotation[2]:.4f}')

        data_np = DVCFlow.rotate_flow(data_np, rotation = np.radians(rotation))

        DeviceManager.garbage()

        return data_np.astype(npdtype)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("--- Example usage ---")
    
    shape = (64, 64, 64)
    indexing = 'dhw'
    # flow_type = 'star'
    # flow_max = 8.0
    # flow = DVCFlowGenerator.generate_synthetic(shape = shape, 
    #                                         indexing = indexing,
    #                                         npdtype = np.float32,
    #                                         type = flow_type, 
    #                                         max_flow = flow_max)
    
    flow_config = {}

    # flow_config["type"] = "star"
    # flow_config["max"] = 16.0
    # flow_config["orders"] = [2, 1, 0]

    # flow_config["type"] = "static"
    # # flow_config["max"] = 20.0
    # flow_config["displacement"] = [12.0, 24.0, 16.0]
    # flow_config["snr_db"] = 16.0
    # flow_config["orders"] = [0, 1, 2]

    # flow_config["type"] = "curve"
    # flow_config["max"] = 20.0
    # flow_config["gamma"] = [1.5, 1.5, 1.5]
    # flow_config["rotation"] = [45.0, 45.0, 45.0]
    # flow_config["slope_ratio"] = 0.8
    # flow_config["slope"] = [12.0, 16.0, 24.0]
    # flow_config["shift"] = [-4.0, -6.0, -8.0]
    # flow_config["orders"] = [0, 1, 2]

    # flow_config["type"] = "sphere"
    # flow_config["max"] = 20.0
    # flow_config["scale"] = [1.0, 1.0, 1.0]
    # flow_config["center"] = [0.5, 0.5, 0.5]
    # flow_config["velocity"] = [0.0, 0.0, 0.0]
    # flow_config["phase_pi"] = 0.5
    # flow_config["orders"] = [0, 1, 2]

    flow_config["type"] = "overall"
    flow_config["max"] = 24.0
    flow_config["snr_db"] = 30.0
    # flow_config["gamma"] = [1.0, 1.0, 1.0]
    # flow_config["scale"] = [1.0, 1.0, 1.0]
    # flow_config["center"] = [0.5, 0.5, 0.5]
    # flow_config["velocity"] = [0.0, 0.0, 0.0]
    # flow_config["phase_pi"] = 0.0
    # flow_config["star_weight"] = 0.0
    # flow_config["star_rotation"] = [60, 45, 30]
    flow_config["star_orders"] = [2, 1, 0]
    flow_config["orders"] = [0, 1, 2]
    # flow_config["rotation"] = [0, 0, 0]

    path_mask = '/asap3/petra3/gpfs/p05/2021/data/11008741/processed/wongtakm/datasets/ki4d4e/dvc/volume_960x1280x1280/2021_11008741_syn0154_103L_Mg5Gd_4w_000_fs402/mask_fill/mask_fill_960x1280x1280.npy'
    mask = DVCMask(data_np=np.load(path_mask), indexing='dhw', npdtype=bool)
    # mask = None

    flow = DVCFlowGenerator.generate_synthetic(shape = shape, 
                                               indexing = indexing,
                                               npdtype = np.float32,
                                               flow_config = flow_config,
                                               mask = mask)
    
    flow_shape = flow.shape
    print(f'synthetic flow.shape = {flow_shape}')

    img0 = flow[0, :, :, flow_shape[3]//2]
    img1 = flow[1, :, :, flow_shape[3]//2]
    img2 = flow[2, :, :, flow_shape[3]//2]

    ### Plot images
    # plt.close('all')

    # fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(18, 6))

    # h0 = ax[0].imshow(img0)
    # fig.colorbar(h0, fraction=0.046, pad=0.04)

    # h1 = ax[1].imshow(img1)
    # fig.colorbar(h1, fraction=0.046, pad=0.04)

    # h2 = ax[2].imshow(img2)
    # fig.colorbar(h2, fraction=0.046, pad=0.04)

    # plt.tight_layout()
    # plt.show()


    ### Plot flows
    _, X, Y, Z = flow.shape
    gx = np.arange(X, dtype = np.float32)
    gy = np.arange(Y, dtype = np.float32)
    gz = np.arange(Z, dtype = np.float32)
    
    # Normalized mesh [0, 1)
    x, y, z = np.meshgrid(gx, gy, gz, sparse=False, indexing='ij')

    x2 = x[:, :, x.shape[2]//2]
    y2 = y[:, :, y.shape[2]//2]
    z2 = z[:, :, z.shape[2]//2]
    u2 = flow.data_np[0, :, :, flow.shape[3]//2]
    v2 = flow.data_np[1, :, :, flow.shape[3]//2]
    w2 = flow.data_np[2, :, :, flow.shape[3]//2]

    plt.close('all')
    # fig = plt.figure(figsize=(10, 10))
    # fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=(12, 12))
    # ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)

    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(18, 6))

    h0 = ax[1].imshow(img0.transpose(), origin='lower')
    fig.colorbar(h0, fraction=0.046, pad=0.04)
    ax[1].set_title(f'dx at z = {flow.shape[3]//2}')
    ax[1].set_xlabel('X axis')
    ax[1].set_ylabel('Y axis')

    h1 = ax[2].imshow(img1.transpose(), origin='lower')
    fig.colorbar(h1, fraction=0.046, pad=0.04)
    ax[2].set_title(f'dy at z = {flow.shape[3]//2}')
    ax[2].set_xlabel('X axis')
    # ax[2].set_ylabel('Y axis')

    # Create the quiver plot
    # This plots arrows to represent the vector field
    # ax.quiver(x, y, z, u, v, w, length=0.01, normalize=True)
    # ax[0].quiver(z2, y2, w2, v2, scale=(1/0.0004), scale_units='height', width=0.0015)
    # ax[0].set_xlabel('Z axis')
    # ax[0].set_ylabel('Y axis')
    # ax[0].quiver(x2, y2, u2, v2, scale=(1/0.0004), scale_units='height', width=0.0015)
    
    # ax[0].quiver(x2, y2, u2, v2, scale=(1/0.0008), scale_units='height', width=0.002, pivot='mid')
    ax[0].streamplot(x2.transpose(), 
                     y2.transpose(), 
                     u2.transpose(), 
                     v2.transpose(), 
                     density=3.0, linewidth=0.8, arrowsize=0.5)
    ax[0].set_aspect('equal', adjustable='box')  # Set aspect ratio to equal
    ax[0].set_xlabel('X axis')
    ax[0].set_ylabel('Y axis')
    ax[0].set_title(f'Slice of 3D flow at z = {flow.shape[3]//2}')
    ax[0].set_xlim(x2.min(), x2.max())
    ax[0].set_ylim(y2.min(), y2.max())

    # ax[1].streamplot(y2, x2, v2, u2, density=5.0, linewidth=0.8, arrowsize=0.5)
    # ax[1].set_xlabel('Y axis')
    # ax[1].set_ylabel('X axis')
    # ax[1].set_title('Streamplot')

    # Show the plot
    plt.tight_layout()
    plt.show()
    