
import torch
import torch.utils.data as data
import torch.nn.functional as F

class CrossSensorDatasetWrapper(data.Dataset):
    def __init__(self, dataset, sensor_id, max_taxels=368, max_channels=25, sequence_length=10):
        self.dataset = dataset
        self.sensor_id = sensor_id
        self.max_taxels = max_taxels
        self.max_channels = max_channels
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        sample = self.dataset[index]
        sensor_data = sample["sensor"]
        sensor_poses = sample["sensor_poses"]
        # print(sensor_poses.shape, sensor_data.shape)
        
        # Ensure (T, N, C)
        if sensor_data.dim() == 2: # (N, C) -> (1, N, C)
             sensor_data = sensor_data.unsqueeze(0)
             
        # Initialize valid_mask based on original data shape
        valid_mask = torch.ones_like(sensor_data) # (T_orig, N_orig, C_orig)

        # Handle sequence length mismatch
        # valid_mask = torch.ones(self.sequence_length) # Removed old 1D mask
        if sensor_data.shape[0] > self.sequence_length:
            # For simplicity, taking last 'sequence_length' frames
            sensor_data = sensor_data[-self.sequence_length:]
            valid_mask = valid_mask[-self.sequence_length:]
        elif sensor_data.shape[0] < self.sequence_length:
             # Pre-pad with zeros
             pad_len = self.sequence_length - sensor_data.shape[0]
             # (T, N, C) -> pad T at front: (0, 0, 0, 0, pad_len, 0)
             sensor_data = F.pad(sensor_data, (0, 0, 0, 0, 0, pad_len))
             valid_mask = F.pad(valid_mask, (0, 0, 0, 0, 0, pad_len))
             # valid_mask[-pad_len:] = 0 # Handled by F.pad default value 0

        t, n, c = sensor_data.shape
                
        # Pad taxels
        if n < self.max_taxels:
            pad_n = self.max_taxels - n
            # We want to pad N: (0, pad_n)
            sensor_data = F.pad(sensor_data, (0, 0, 0, pad_n, 0, 0))
            valid_mask = F.pad(valid_mask, (0, 0, 0, pad_n, 0, 0))
        elif n > self.max_taxels:
             sensor_data = sensor_data[:, :self.max_taxels, :]
             valid_mask = valid_mask[:, :self.max_taxels, :]

        # Pad channels
        if c < self.max_channels:
            pad_c = self.max_channels - c            
            # C: 0, pad_c
            # N: 0, 0
            # T: 0, 0
            sensor_data = F.pad(sensor_data, (0, pad_c, 0, 0, 0, 0))
            valid_mask = F.pad(valid_mask, (0, pad_c, 0, 0, 0, 0))
            
        elif c > self.max_channels:
             sensor_data = sensor_data[:, :, :self.max_channels]
             valid_mask = valid_mask[:, :, :self.max_channels]

        # Process sensor_poses
        # Ensure (T, N, 7) - typical pose dimension
        if sensor_poses.dim() == 2:
            sensor_poses = sensor_poses.unsqueeze(0)
            
        # Handle sequence length mismatch for poses
        if sensor_poses.shape[0] > self.sequence_length:
            sensor_poses = sensor_poses[-self.sequence_length:]
        elif sensor_poses.shape[0] < self.sequence_length:
             pad_len = self.sequence_length - sensor_poses.shape[0]
             # (T, N_p, C_p) -> pad T at front
             sensor_poses = F.pad(sensor_poses, (0, 0, 0, 0, 0, pad_len))

        # Pad taxels for poses
        n_poses = sensor_poses.shape[1]
        if n_poses < self.max_taxels:
            pad_n_poses = self.max_taxels - n_poses
            # Pad N dim (dim 1) -> (0, 0, 0, pad_n, 0, 0)
            sensor_poses = F.pad(sensor_poses, (0, 0, 0, pad_n_poses, 0, 0))
        elif n_poses > self.max_taxels:
             sensor_poses = sensor_poses[:, :self.max_taxels, :]


        new_sample = {}
        new_sample["sensor"] = sensor_data
        new_sample["sensor_poses"] = sensor_poses
        new_sample["sensor_ids"] = torch.tensor(self.sensor_id, dtype=torch.long)
        new_sample["valid_mask"] = valid_mask
        return new_sample

    def __len__(self):
        return len(self.dataset)
        
    def __getattr__(self, name):
        return getattr(self.dataset, name)
