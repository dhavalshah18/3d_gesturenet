import pathlib
import numpy as np
import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import misc as ms

class GestureData(data.Dataset):
    """
    Class defined to handle MRA data
    derived from pytorch's Dataset class.
    """

    def __init__(self, root_path, mode="", z=10):
        # Note: Should already be split into train, val and test folders

        self.root_dir = pathlib.Path(root_path)

        self.images_dir = self.root_dir.joinpath("images/")
        
        self.z = z

        # Read split .txt file
        split_file = self.root_dir.joinpath(mode+".txt")
        with open(str(split_file), "r") as file:
            self.sequence_files = file.readlines()

    def __len__(self):
        return len(self.sequence_files)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            return [self[i] for i in range(index)]
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index > len(self):
                raise IndexError("The index (%d) is out of range." % index)

            # get the data from direct index
            return self.get_item_from_index(index)

        else:
            raise TypeError("Invalid argument type.")

    def get_item_from_index(self, index):
        seq_file, label = self.sequence_files[index].strip().split(" ")

        with open(seq_file, "r") as file:
            images = file.readlines()
        
        temp = PIL.Image.open(images[0].strip())
        temp = transforms.Resize(150)(temp)
        temp = np.array(temp)
        tensor_size = (self.z, ) + (temp.shape[-1], ) + (temp.shape[0:-1])

        img_sequence = torch.empty(tensor_size, dtype=torch.float)

        for i in range(self.z):
            img = PIL.Image.open(images[i].strip())
            img = transforms.Resize(150)(img)
            
            img_tensor = transforms.ToTensor()(img)
#             std, mean = torch.std_mean(img_tensor, dim=[1, 2])
#             img_tensor = transforms.Normalize(mean, std)(img_tensor)
            img_sequence[i] = img_tensor

        img_sequence = img_sequence.permute(1, 2, 3, 0)
        std, mean = torch.std_mean(img_sequence)
        normalize = ms.Normalize(mean, std)
        img_sequence = normalize(img_sequence)
        # -1 for label as our labels are 1-35
        # but we want 0-34
        label = float(label) - 1.

        return img_sequence, label
