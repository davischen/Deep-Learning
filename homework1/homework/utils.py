from PIL import Image
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        self.dataset_path=dataset_path
        with open(self.dataset_path+"/labels.csv", newline='') as csvFile:
          self.datalist=list(csv.reader(csvFile))
          self.datalist.pop(0)
        #raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        return len(self.datalist)
        #raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        image_filename,image_label = self.datalist[idx][:2]
        I = Image.open(self.dataset_path+"/"+image_filename)
        #convert ndarray to tensor
        image_to_tensor = transforms.ToTensor()
        #print(image_tensor.shape)
        return (image_to_tensor(I),LABEL_NAMES.index(image_label))
        #raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
