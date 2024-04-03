import torch
import torch.nn as nn
import numpy as np

# from .VolRAFT.dvc_loss_volraft import DVCLossRAFT

class CriterionFactory:
    @staticmethod
    def build_instance(config, device, ptdtype, flow_margin_predict, flow_margin_target) -> nn.Module:
        # L1 or L2 loss
        if str(config["loss_type"]).lower() in ('l1', 'l2'):
            criterion = DVCLoss(config = config,
                                device = device,
                                ptdtype = ptdtype,
                                margin_predict = flow_margin_predict,
                                margin_target = flow_margin_target)
            print(f'Loss: type = {config["loss_type"]}')

        # L1 or L2 loss
        if str(config["loss_type"]).lower() == 'raftloss':
            criterion = DVCLossRAFT(config = config,
                                    device = device,
                                    ptdtype = ptdtype)
            print(f'Loss: type = {config["loss_type"]}')

        return criterion
    
def epe(predict, target, axis=1, mask=None):
    """
    Calculate the End-Point Error (EPE) for optical flow with a specified axis and an optional mask.

    :param predict: Predicted optical flow (Tensor or NumPy array)
    :param target: Ground truth optical flow (Tensor or NumPy array)
    :param axis: Axis along which the flow field is represented (default is 1)
    :param mask: Optional binary mask (Tensor or NumPy array) to indicate valid positions
    :return: EPE (float), EPE_MAP (NumPy Array)
    """

    if isinstance(predict, torch.Tensor) and isinstance(target, torch.Tensor):
        # PyTorch computation
        difference = target - predict
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError("For PyTorch tensors, mask must also be a PyTorch tensor")
            # difference = difference * mask.bool()
        
        distance = torch.sqrt(torch.sum(difference ** 2, dim=axis))
        epe_value = torch.mean(distance[mask.bool() == 1]).item() if mask is not None else torch.mean(distance).item()
        epe_map = distance.numpy()

    elif isinstance(predict, np.ndarray) and isinstance(target, np.ndarray):
        # NumPy computation
        difference = target - predict
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise TypeError("For NumPy arrays, mask must also be a NumPy array")
            # difference = difference * mask.astype(bool)
        
        distance = np.sqrt(np.sum(difference ** 2, axis=axis))
        epe_value = np.mean(distance[mask.astype(bool) == 1]) if mask is not None else np.mean(distance)
        epe_map = distance
        
    else:
        raise TypeError("Input types must be both PyTorch Tensors or both NumPy arrays")

    return epe_value, epe_map

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=400.0):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim = 1, keepdim = True).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

# Loss function
class DVCLoss(nn.Module):
    def __init__(self, 
                 config,
                 device,
                 ptdtype,
                 margin_predict = [(0, 0, 0), (0, 0, 0)], 
                 margin_target = [(0, 0, 0), (0, 0, 0)]):
        super(DVCLoss, self).__init__()

        # Define the device and data type
        self.device = device
        self.ptdtype = ptdtype

        # Define the margin for prediction
        # margin represent the number of pixel at the borders
        # the number of pixel to be shrinked is (2 * margin)
        self.margin_p_start = margin_predict[0]
        self.margin_p_end = margin_predict[1]

        # Define the margin for target
        # margin represent the number of pixel at the borders
        # the number of pixel to be shrinked is (2 * margin)
        self.margin_t_start = margin_target[0]
        self.margin_t_end = margin_target[1]

        # Define the loss module
        # self.loss_module = self.get_loss_module(config)
        self.data_term = DataLoss(config = config)

        if "loss_tvl1_weight" in config.keys():
            self.reg_tv = TVL1Loss(device = self.device,
                                ptdtype = self.ptdtype,
                                weight = float(config["loss_tvl1_weight"]))
            
            print(f'Use TV-L1 regularization with weight: {config["loss_tvl1_weight"]}')
        else:
            self.reg_tv = None

        return

    def get_loss_module(self, config):
        if str(config["loss_type"]).lower() == "l2":
            loss_module = nn.MSELoss(reduction='sum')

        if str(config["loss_type"]).lower() == "l1":
            loss_module = nn.SmoothL1Loss(reduction='sum')

        print(f'Select loss: {config["loss_type"]}')  

        return loss_module

    def forward(self, predict, target, mask=None):
        # Get the shape of prediction
        _, nc_p, nd_p, nh_p, nw_p = predict.shape

        # Get the region-of-interest for prediction
        predict_roi = predict[:, :, 
                              self.margin_p_start[0]:nd_p-self.margin_p_end[0],
                              self.margin_p_start[1]:nh_p-self.margin_p_end[1], 
                              self.margin_p_start[2]:nw_p-self.margin_p_end[2]
                              ]

        # Get the shape of target
        _, nc_t, nd_t, nh_t, nw_t = target.shape

        # Get the region-of-interest for target
        target_roi = target[:, :, 
                            self.margin_t_start[0]:nd_t-self.margin_t_end[0],
                            self.margin_t_start[1]:nh_t-self.margin_t_end[1], 
                            self.margin_t_start[2]:nw_t-self.margin_t_end[2]
                            ]
        
        loss_reg = None

        if mask is None:
            # Calculate the count of valid pixels
            count = (predict_roi.shape[1] * predict_roi.shape[2] * predict_roi.shape[3] * predict_roi.shape[4]).astype(float)

            # loss_sum = self.loss_module(predict_roi, target_roi)

            # data term
            loss_data = self.data_term(predict_roi, target_roi) / count

            # regularization term
            if self.reg_tv is not None:
                loss_reg = self.reg_tv(predict_roi, 1.0) / count
        else:
            # Get the shape of mask
            _, nc_m, _, _, _ = mask.shape

            # Find the scaling factor for the mask
            if (nc_p == nc_t):
                mask_factor = nc_p / nc_m
            else:
                mask_factor = 1.0

            # Find the region of interest
            mask_roi = mask[:, :, 
                            self.margin_p_start[0]:nd_p-self.margin_p_end[0],
                            self.margin_p_start[1]:nh_p-self.margin_p_end[1], 
                            self.margin_p_start[2]:nw_p-self.margin_p_end[2]
                            ]
            
            mask_roi.requires_grad_(False)
            
            # Calculate the count of valid pixels in mask
            count = (torch.sum(mask_roi) * mask_factor).requires_grad_(False)

            # Calculate loss_sum
            # loss_sum = self.loss_module(predict_roi * mask_roi, target_roi * mask_roi)

            # data term
            loss_data = self.data_term(predict_roi * mask_roi, target_roi * mask_roi) / count

            # regularization term
            if self.reg_tv is not None:
                loss_reg = self.reg_tv(predict_roi, mask_roi) / count

        # Record the loss
        with torch.no_grad():
            if loss_reg is not None:
                loss_info = [loss_data.item(), loss_reg.item()]
            else:
                loss_reg = 0.0
                loss_info = [loss_data.item(), 0.0]
        
        return loss_data + loss_reg, loss_info
        
class DataLoss(nn.Module):
    def __init__(self, config):
        super(DataLoss, self).__init__()

        # Define the loss module
        self.loss_module = self.get_loss_module(config)

    def get_loss_module(self, config):
        if str(config["loss_type"]).lower() == "l2":
            loss_module = nn.MSELoss(reduction='sum')

        if str(config["loss_type"]).lower() == "l1":
            loss_module = nn.SmoothL1Loss(reduction='sum')

        print(f'Select loss: {config["loss_type"]}')

        return loss_module
    
    def forward(self, predict, target):
        return self.loss_module(predict, target)
    
# Loss function
class DVCLossRAFT(nn.Module):
    def __init__(self, 
                 config,
                 device,
                 ptdtype):
        super(DVCLossRAFT, self).__init__()

        self.max_flow = config["proj_max_flow"]

        self.device = device

        self.ptdtype = ptdtype,

        self.config = config

    def forward(self, predict, target, mask=None):
        loss, metrics = sequence_loss(predict, 
                                      target, 
                                      mask, 
                                      gamma = self.config["loss_gamma"],
                                      max_flow= self.config["proj_max_flow"])
        
        loss_info = [metrics['epe'], metrics['1px'], metrics['3px'], metrics['5px']]
        
        return loss, loss_info

# Regularization by Total Variation L1 (TVL1)
class TVL1Loss(nn.Module):
    def __init__(self, device, ptdtype, weight=None):
        super(TVL1Loss, self).__init__()
        self.weight = weight
        self.ptdtype = ptdtype
        self.device = device

    def forward(self, predict, mask):
        # Prepare the horizontal, vertical and depth total variation field
        h_tv = torch.zeros_like(predict, device = self.device, dtype = self.ptdtype)
        v_tv = torch.zeros_like(predict, device = self.device, dtype = self.ptdtype)
        d_tv = torch.zeros_like(predict, device = self.device, dtype = self.ptdtype)

        # horizontal TV (or x-axis)
        h_tv[:, :, :-1, :, :] = torch.abs(predict[:, :, :-1, :, :] - predict[:, :, 1:, :, :])

        # vertical TV
        v_tv[:, :, :, :-1, :] = torch.abs(predict[:, :, :, :-1, :] - predict[:, :, :, 1:, :])
        
        # depth TV
        d_tv[:, :, :, :, :-1] = torch.abs(predict[:, :, :, :, :-1] - predict[:, :, :, :, 1:])

        if self.weight is None:
            return torch.sum(h_tv * mask) + torch.sum(v_tv * mask) + torch.sum(d_tv * mask)
        else:
            return self.weight * (torch.sum(h_tv * mask) + torch.sum(v_tv * mask) + torch.sum(d_tv * mask))

if __name__ == "__main__":
    print("--- Example usage ---")
    