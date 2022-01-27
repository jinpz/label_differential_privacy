import torchvision.transforms as transforms
from lib.dataset.cifar import random_subset
from torchvision.datasets import MNIST
from lib.dataset.randaugment import RandAugmentMC

mnist_mean = (0.1307,)
mnist_std = (0.3081,)


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=28, padding=4, padding_mode="reflect"
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=28, padding=4, padding_mode="reflect"
                ),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


TRANSFORM_LABELED_MNIST = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_std),
    ]
)

TRANSFORM_UNLABELED_MNIST = TransformFixMatch(mean=mnist_mean, std=mnist_std)

TRANSFORM_TEST_MNIST = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mnist_mean, std=mnist_std)]
)


def get_mnist10(root: str, student_dataset_max_size: int, student_seed: int):
    labeled_dataset = MNIST(
        root=root, train=True, download=True, transform=TRANSFORM_LABELED_MNIST
    )
    test_dataset = MNIST(
        root=root, train=False, download=True, transform=TRANSFORM_TEST_MNIST
    )
    unlabeled_dataset = MNIST(
        root=root, train=True, download=True, transform=TRANSFORM_UNLABELED_MNIST
    )
    student_dataset = random_subset(
        dataset=MNIST(
            root=root, train=True, download=True, transform=TRANSFORM_LABELED_MNIST
        ),
        n_samples=student_dataset_max_size,
        seed=student_seed,
    )

    return {
        "labeled": labeled_dataset,
        "unlabeled": unlabeled_dataset,
        "test": test_dataset,
        "student": student_dataset,
    }



