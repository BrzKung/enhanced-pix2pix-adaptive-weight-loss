import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

# Import from modules
from test_config import DEVICE, LEARNING_RATE
from dataset import MapDataset
from models import Generator
from utils import load_checkpoint
from test_utils import save_some_examples_resized
from metrics import evaluate_all_metrics, print_metrics

checkpoint = 1000
file_path = f"./result/pix2pix_original/recent/{checkpoint}_epochs"
checkpoint_path = f"./checkpoints/pix2pix_original/recent/gen_{checkpoint}.pth.tar"
TEST_DIR = f"./datasets/test"
history_file_name = f"/history_{checkpoint}.csv"

# Plot training history
try:
    history_df = pd.read_csv(f"{file_path}{history_file_name}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["epoch"],
        history_df["total_loss"],
        label="Generator Loss",
        color="red",
        linestyle="-",
    )
    plt.plot(
        history_df["epoch"],
        history_df["discriminator_loss"],
        label="Discriminator Loss",
        color="purple",
        linestyle="-",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}/graph_1.jpg", format="jpg", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["epoch"],
        history_df["l1_loss"],
        label="L1 Loss",
        color="red",
        linestyle="-",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("L1 Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}/graph_3.jpg", format="jpg", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_df["epoch"],
        history_df["l1_loss"],
        label="L1 Total Loss",
        color="red",
        linestyle="-",
    )
    plt.plot(
        history_df["epoch"],
        history_df["generator_loss"],
        label="Adversarial Loss",
        color="purple",
        linestyle="-",
    )
    plt.plot(
        history_df["epoch"],
        history_df["total_loss"],
        label="Generator Loss",
        color="orange",
        linestyle="-",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Combined Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}/graph_4.jpg", format="jpg", dpi=300)
    plt.show()
except Exception as e:
    print(f"Could not load history file: {e}")

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

real_list = []
result_list = []

while True:
    try:
        x, y, y_fake, mask, inpainted = save_some_examples_resized(
            test_gen_model,
            iter_test_loader,
            folder=f"{file_path}",
            run_num=testing_num,
            device=DEVICE,
            resize_size=512,
        )

        resize = transforms.Resize((299, 299))
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
