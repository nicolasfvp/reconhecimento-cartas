"""Carregamento de dados e transformacoes (PyTorch).

Espera a estrutura do dataset gpiosenka "Cards Image Dataset-Classification":

    data_dir/
      train/   <53 subpastas, 1 por classe>
      valid/   (ou 'validation' / 'val')
      test/

Cada subpasta contem as imagens daquela classe. Os nomes das classes sao
derivados automaticamente das subpastas (nao ha rotulos hardcoded).
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import IMAGENET_MEAN, IMAGENET_STD

# Nomes alternativos aceitos para cada split (datasets variam a nomenclatura).
_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "valid": ("valid", "validation", "val"),
    "test": ("test", "testing"),
}


def _resolve_split_dir(data_dir: str | Path, split: str) -> Path:
    """Encontra a pasta do split aceitando nomes alternativos."""
    base = Path(data_dir)
    for alias in _SPLIT_ALIASES[split]:
        candidate = base / alias
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Nao encontrei a pasta do split '{split}' em {base}. "
        f"Esperado um destes nomes: {_SPLIT_ALIASES[split]}."
    )


def build_transforms(img_size: int = 224, augment: bool = True):
    """Cria as transformacoes de treino (com/sem augmentation) e de avaliacao."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if augment:
        train_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                # Augmentation leve adequada a cartas: rotacao, brilho/contraste,
                # leve translacao/zoom e oclusao parcial (RandomErasing).
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
            ]
        )
    else:
        train_tf = eval_tf

    return train_tf, eval_tf


def build_dataloaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    augment: bool = True,
):
    """Constroi os DataLoaders de train/valid/test.

    Returns:
        (loaders, class_names) onde loaders e um dict com chaves
        'train', 'valid', 'test' e class_names e a lista ordenada de classes.
    """
    train_tf, eval_tf = build_transforms(img_size=img_size, augment=augment)

    train_ds = datasets.ImageFolder(_resolve_split_dir(data_dir, "train"), transform=train_tf)
    valid_ds = datasets.ImageFolder(_resolve_split_dir(data_dir, "valid"), transform=eval_tf)
    test_ds = datasets.ImageFolder(_resolve_split_dir(data_dir, "test"), transform=eval_tf)

    # Sanidade: as classes devem ser as mesmas (mesma ordem) nos tres splits.
    if not (train_ds.classes == valid_ds.classes == test_ds.classes):
        raise ValueError(
            "As classes diferem entre os splits train/valid/test. "
            "Verifique se as subpastas (uma por classe) sao identicas em cada split."
        )

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        ),
        "valid": DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        ),
    }
    return loaders, train_ds.classes


def build_eval_loader(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """DataLoader de avaliacao para uma pasta arbitraria (ex.: teste OOD).

    A pasta deve conter 1 subpasta por classe (mesmos nomes do treino).
    Returns:
        (loader, dataset) — o dataset expoe .classes e .samples.
    """
    _, eval_tf = build_transforms(img_size=img_size, augment=False)
    ds = datasets.ImageFolder(data_dir, transform=eval_tf)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return loader, ds
