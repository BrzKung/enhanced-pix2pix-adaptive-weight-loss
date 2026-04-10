import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Import from modules
from test_config import DEVICE, LEARNING_RATE
from dataset import MapDataset
from models import Generator
from utils import load_checkpoint
from test_utils import save_some_examples
from metrics import evaluate_all_metrics, print_metrics

checkpoint = 4849
file_path = f"./result/pix2pix_fixed_loss/recent/{checkpoint}_epochs"
checkpoint_path = f"./checkpoints/pix2pix_fixed_loss/recent/gen_{checkpoint}.pth.tar"
TEST_DIR = f"./datasets/test"

# Initialize test dataset and loader
test_dataset = MapDataset(root_dir=TEST_DIR, masked=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load generator model
test_gen_model = Generator(in_channels=3, features=64).to(DEVICE)
test_opt_gen = optim.Adam(
    test_gen_model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)
load_checkpoint(f"{checkpoint_path}", test_gen_model, test_opt_gen, LEARNING_RATE)

# Generate and collect results
testing_num = 1
iter_test_loader = iter(test_loader)
resize = transforms.Resize(
    (299, 299), interpolation=transforms.InterpolationMode.BICUBIC
)

real_list = []
result_list = []

while True:
    try:
        x, y, y_fake, mask, inpainted = save_some_examples(
            test_gen_model,
            iter_test_loader,
            folder=f"{file_path}",
            run_num=testing_num,
            device=DEVICE,
        )

        result = resize(inpainted)
        real = resize(y)

        result_list.append(result)
        real_list.append(real)

        testing_num += 1
    except StopIteration:
        break

# Evaluate metrics
result_images = torch.cat(result_list, dim=0)
real_images = torch.cat(real_list, dim=0)

metrics = evaluate_all_metrics(real_images, result_images, device=DEVICE)
print_metrics(metrics)
