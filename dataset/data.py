import dataclasses
import itertools
import operator
import os
import pickle
import random
import shutil
import sys
import time
from abc import abstractmethod
from enum import auto
from itertools import groupby
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Sequence, Set, Tuple, Union
from Bio import SeqIO
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from plotly.graph_objs.layout import Shape
from strenum import StrEnum
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
from matplotlib.patches import Rectangle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, MAX_RNA_SIZE

AugmentSpec = Mapping[str, Any]
Interaction = Mapping[str, Any]


class Interaction(dict):
    def get_boundaries(self, gene_id: str) -> Tuple[int, int]:
        if gene_id == self["gene1"]:
            return self["x1"], self["x1"] + self["w"]
        elif gene_id == self["gene2"]:
            return self["y1"], self["y1"] + self["h"]
        raise RuntimeError(f"Gene {gene_id} not found in interaction {self}")


Sample = Mapping[str, Any]
AugmentResult = Mapping[str, Any]


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclasses.dataclass(frozen=True)
class BBOX:
    x1: int
    x2: int
    y1: int
    y2: int

    @classmethod
    def from_xyhw(cls, x: int, y: int, height: int, width: int) -> "BBOX":
        return BBOX(
            x1=x,
            x2=x + width,
            y1=y,
            y2=y + height,
        )

    @classmethod
    def from_interaction(cls, interaction) -> "BBOX":
        return BBOX.from_xyhw(
            x=int(interaction["x1"]),
            y=int(interaction["y1"]),
            width=int(interaction["w"]),
            height=int(interaction["h"]),
        )

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def coords(self):
        return self.x1, self.x2, self.y1, self.y2


@dataclasses.dataclass(frozen=True)
class Sample:
    """
    A sample is a dictionary containing the following keys:
    - gene1: name of the first gene
    - gene2: name of the second gene
    - bbox: bounding box of the augmented sample
    - couple_id: id of the couple
    - policy: name of the augmentation policy used
    - interacting: whether the sample contains an interaction or not
    - interaction_bbox: bounding box of the interaction used for augmentation
    - all_couple_interactions: list of all interactions between the two genes
    - gene1_info: information about the first gene
    """

    gene1: str
    gene2: str
    bbox: BBOX
    couple_id: str
    policy: str
    interacting: bool
    seed_interaction_bbox: BBOX
    all_couple_interactions: Sequence[Interaction]
    gene1_info: Mapping[str, Any]
    gene2_info: Mapping[str, Any]


class AugmentMode(StrEnum):
    EASY_POS = auto()
    EASY_NEG = auto()
    HARD_POS = auto()
    HARD_NEG = auto()


class InteractionSelectionPolicy(StrEnum):
    RANDOM_ONE = auto()
    LARGEST = auto()
    SMALLER = auto()
    ALL = auto()


class AugmentPolicy:
    def __init__(
        self,
        per_sample: Union[float, int],
        width_probabilities: Union[
            Mapping[float, float], Mapping[Tuple[int, int], float]
        ],
        height_probabilities: Union[
            Mapping[float, float], Mapping[Tuple[int, int], float]
        ],
        interacting: Collection[bool],
        max_size: int = MAX_RNA_SIZE,
    ):
        """

        :param per_sample: if float, it is the probability to apply the augmentation to a given sample. If integer,
        it is the multiplier to compute the number of samples generated from the starting one.
        :param width_probabilities: if the keys are floats, they are interpreted as multipliers. If they are tuples
        of ints, they are interpreted as windows to draw from.
        :param height_probabilities: if the keys are floats, they are interpreted as multipliers. If they are tuples
        of ints, they are interpreted as windows to draw from.
        :param interacting: boolean specifying if this augmentation can be applied to samples having an interaction
        or not
        """
        assert per_sample > 0
        self.per_sample: Union[float, int] = per_sample  # probability or multiplier

        self.width_bins = list(width_probabilities.keys())
        self.width_probabilities = torch.tensor(list(width_probabilities.values()))

        self.height_bins = list(height_probabilities.keys())
        self.height_probabilities = torch.tensor(list(height_probabilities.values()))
        self.interacting: Set[bool] = set(interacting)

        self.max_size: int = max_size

    @property
    def name(self) -> str:
        mapping = {
            "EasyPosAugment": "easypos",
            "HardPosAugment": "hardpos",
            "EasyNegAugment": "easyneg",
            "HardNegAugment": "hardneg",
            "RegionSpecNegAugment": "regionneg",
            "SmartNegAugment": "smartneg",
        }
        return mapping[type(self).__name__]

    @abstractmethod
    def augment(
        self,
        target_interaction: int,
        couple_interactions: Sequence[Interaction],
        gene1: str,
        gene2: str,
        dataset: "RNADataset",
    ) -> AugmentResult:
        """
        Augment an interaction between a couple of genes.

        param target_interaction: index of the interaction to augment
        param couple_interactions: list of interactions between the two genes
        param gene1: name of the first gene
        param gene2: name of the second gene
        param dataset: dataset from which the sample is drawn

        return: a dictionary containing the augmented sample
        """
        raise NotImplementedError

    def get_target_interactions(
        self, couple_interactions: Sequence[Interaction]
    ) -> List[int]:
        return list(range(len(couple_interactions)))

    def generate_augment_specs(
        self, couple_interactions: Sequence[Interaction]
    ) -> Sequence[AugmentSpec]:
        #print(couple_interactions[0]) -> {'couple': 'ENSG00000000419_ENSG00000249150', 'gene1': 'ENSG00000000419', 'gene2': 'ENSG00000249150', 'interacting': False, 'length_1': 1161, ...}
        interacting = set(
            interaction["interacting"] for interaction in couple_interactions
        ) 
        assert len(interacting) == 1
        if len(set.intersection(interacting, self.interacting)) == 0:
            return []

        # probability
        if 0 < self.per_sample < 1:
            if random.random() >= self.per_sample:
                return []
            else:
                multiplier: int = 1
        # multiplier
        else:
            assert isinstance(self.per_sample, int)
            multiplier = self.per_sample

        augment_specs = [
            dict(
                augment_policy=self,
                couple_interactions=couple_interactions,
                target_interaction=target_interaction,
            )
            for target_interaction in self.get_target_interactions(
                couple_interactions=couple_interactions
            )
            * multiplier
        ]

        return [
            augment_spec for augment_spec in augment_specs if augment_spec is not None
        ]


class EasyPosAugment(AugmentPolicy):
    def __init__(
        self,
        interaction_selection: InteractionSelectionPolicy,
        per_sample: Union[float, int],
        width_multipliers: Mapping[float, float],
        height_multipliers: Mapping[float, float],
    ) -> None:
        super().__init__(
            per_sample=per_sample,
            width_probabilities=width_multipliers,
            height_probabilities=height_multipliers,
            interacting=[True],
        )
        self.interaction_selection: InteractionSelectionPolicy = interaction_selection

    def get_target_interactions(
        self, couple_interactions: Sequence[Interaction]
    ) -> Sequence[int]:
        #print(couple_interactions) -> [{'couple': 'ENSG00000000419_ENSG00000280193', 'gene1': 'ENSG00000000419', 'gene2': 'ENSG00000280193', 'interacting': True, 'length_1': 1161, 'length_2': 2327, 'protein_coding_1': True, 'protein_coding_2': False, 'x1': 543, 'y1': 1294, 'w': 30, 'h': 44, 'matrix_area': 2701647, 'interaction_area': 1320}]
        if self.interaction_selection == InteractionSelectionPolicy.ALL:
            target_interactions = list(range(len(couple_interactions)))
        elif self.interaction_selection == InteractionSelectionPolicy.RANDOM_ONE:
            target_interactions = [random.randint(0, len(couple_interactions) - 1)]
        elif self.interaction_selection == InteractionSelectionPolicy.LARGEST:
            target_interactions = [
                sorted(
                    enumerate(couple_interactions),
                    key=lambda index2interaction: index2interaction[1][
                        "interaction_area"
                    ],
                )[-1][0]
            ]

        elif self.interaction_selection == InteractionSelectionPolicy.SMALLER:
            target_interactions = [
                sorted(
                    enumerate(couple_interactions),
                    key=lambda index2interaction: index2interaction[1][
                        "interaction_area"
                    ],
                )[0][0]
            ]
        else:
            raise NotImplementedError

        return target_interactions

    def augment(
        self,
        target_interaction: int,
        couple_interactions: Sequence[Interaction],
        gene1: str,
        gene2: str,
        dataset: "RNADataset",
    ) -> AugmentResult:
        """
        Augment a positive sample by sampling a new width and height for the target interaction.
        :param target_interaction: index of the interaction to augment
        :param couple_interactions: list of interactions for the couple
        :param gene1: name of the first gene
        :param gene2: name of the second gene
        :param dataset: dataset containing the gene info"""
        # Get the target interaction
        interaction: Interaction = couple_interactions[target_interaction]

        # Get the gene info
        gene1_info: Mapping[str, Any] = dataset.gene2info[gene1]
        gene2_info: Mapping[str, Any] = dataset.gene2info[gene2]

        # gene1 refers to dim1 of the matrix, so to the width. Symmetrically, gene2 refers to the height (dim 0)
        gene1_length: int = len(gene1_info["cdna"])
        gene2_length: int = len(gene2_info["cdna"])

        # First, sample the target width & height from their distributions
        # TODO: limit lengths to not intersect with other interactions?
        target_width: int = _sample_target_dimension_multiplier(
            bins=self.width_bins,
            probabilities=self.width_probabilities,
            max_size=min(gene1_length, self.max_size),
            starting_dim=interaction["w"],
        )
        target_height: int = _sample_target_dimension_multiplier(
            bins=self.height_bins,
            probabilities=self.height_probabilities,
            max_size=min(gene2_length, self.max_size),
            starting_dim=interaction["h"],
        )

        # We can now crop each dimension separately.
        interaction_x1: int = int(interaction["x1"])
        interaction_x2: int = int(interaction_x1 + interaction["w"])
        target_x1 = _compute_target(
            interaction_p1=interaction_x1,
            interaction_p2=interaction_x2,
            min_overlap=1,
            max_length=gene1_length,
            interaction_length=interaction["w"],
            target_length=target_width,
        )
        target_x2 = min(target_x1 + target_width, gene1_length)

        interaction_y1: int = int(interaction["y1"])
        interaction_y2: int = int(interaction_y1 + interaction["h"])
        target_y1 = _compute_target(
            interaction_p1=interaction_y1,
            interaction_p2=interaction_y2,
            min_overlap=1,
            max_length=gene2_length,
            interaction_length=interaction["h"],
            target_length=target_height,
        )
        target_y2 = min(target_y1 + target_height, gene2_length)

        return dict(
            bbox=BBOX(
                x1=int(target_x1),
                x2=int(target_x2),
                y1=int(target_y1),
                y2=int(target_y2),
            ),
            gene1=gene1,
            gene2=gene2,
            interacting=True,
        )


class SmartNegAugment(AugmentPolicy):
    def __init__(
        self,
        interaction_selection: InteractionSelectionPolicy,
        per_sample: Union[float, int],
        width_multipliers: Mapping[float, float],
        height_multipliers: Mapping[float, float],
    ) -> None:
        super().__init__(
            per_sample=per_sample,
            width_probabilities=width_multipliers,
            height_probabilities=height_multipliers,
            interacting=[False],  #only difference from EasyPosAugment
        )
        self.interaction_selection: InteractionSelectionPolicy = interaction_selection

    def get_target_interactions(
        self, couple_interactions: Sequence[Interaction]
    ) -> Sequence[int]:
        if self.interaction_selection == InteractionSelectionPolicy.ALL:
            target_interactions = list(range(len(couple_interactions)))
        elif self.interaction_selection == InteractionSelectionPolicy.RANDOM_ONE:
            target_interactions = [random.randint(0, len(couple_interactions) - 1)]
        elif self.interaction_selection == InteractionSelectionPolicy.LARGEST:
            target_interactions = [
                sorted(
                    enumerate(couple_interactions),
                    key=lambda index2interaction: index2interaction[1][
                        "interaction_area"
                    ],
                )[-1][0]
            ]

        elif self.interaction_selection == InteractionSelectionPolicy.SMALLER:
            target_interactions = [
                sorted(
                    enumerate(couple_interactions),
                    key=lambda index2interaction: index2interaction[1][
                        "interaction_area"
                    ],
                )[0][0]
            ]
        else:
            raise NotImplementedError

        return target_interactions

    def augment(
        self,
        target_interaction: int,
        couple_interactions: Sequence[Interaction],
        gene1: str,
        gene2: str,
        dataset: "RNADataset",
    ) -> AugmentResult:
        """
        Augment a positive sample by sampling a new width and height for the target interaction.
        :param target_interaction: index of the interaction to augment
        :param couple_interactions: list of interactions for the couple
        :param gene1: name of the first gene
        :param gene2: name of the second gene
        :param dataset: dataset containing the gene info"""
        # Get the target interaction
        interaction: Interaction = couple_interactions[target_interaction]

        # Get the gene info
        gene1_info: Mapping[str, Any] = dataset.gene2info[gene1]
        gene2_info: Mapping[str, Any] = dataset.gene2info[gene2]

        # gene1 refers to dim1 of the matrix, so to the width. Symmetrically, gene2 refers to the height (dim 0)
        gene1_length: int = len(gene1_info["cdna"])
        gene2_length: int = len(gene2_info["cdna"])

        # First, sample the target width & height from their distributions
        # TODO: limit lengths to not intersect with other interactions?
        target_width: int = _sample_target_dimension_multiplier(
            bins=self.width_bins,
            probabilities=self.width_probabilities,
            max_size=min(gene1_length, self.max_size),
            starting_dim=interaction["w"],
        )
        target_height: int = _sample_target_dimension_multiplier(
            bins=self.height_bins,
            probabilities=self.height_probabilities,
            max_size=min(gene2_length, self.max_size),
            starting_dim=interaction["h"],
        )

        # We can now crop each dimension separately.
        interaction_x1: int = int(interaction["x1"])
        interaction_x2: int = int(interaction_x1 + interaction["w"])
        target_x1 = _compute_target(
            interaction_p1=interaction_x1,
            interaction_p2=interaction_x2,
            min_overlap=1,
            max_length=gene1_length,
            interaction_length=interaction["w"],
            target_length=target_width,
        )
        target_x2 = min(target_x1 + target_width, gene1_length)

        interaction_y1: int = int(interaction["y1"])
        interaction_y2: int = int(interaction_y1 + interaction["h"])
        target_y1 = _compute_target(
            interaction_p1=interaction_y1,
            interaction_p2=interaction_y2,
            min_overlap=1,
            max_length=gene2_length,
            interaction_length=interaction["h"],
            target_length=target_height,
        )
        target_y2 = min(target_y1 + target_height, gene2_length)

        return dict(
            bbox=BBOX(
                x1=int(target_x1),
                x2=int(target_x2),
                y1=int(target_y1),
                y2=int(target_y2),
            ),
            gene1=gene1,
            gene2=gene2,
            interacting=False, #The other difference with respect to EasyPosAugment
        )
    
class HardPosAugment(AugmentPolicy):
    def __init__(
        self,
        interaction_selection: InteractionSelectionPolicy,
        per_sample: Union[float, int],
        width_multipliers: Mapping[float, float],
        height_multipliers: Mapping[float, float],
        min_width_overlap: float,
        min_height_overlap: float,
    ):
        """
        interaction_selection: how to select the interaction to augment
        per_sample: how many interactions to augment per sample
        width_multipliers: mapping from width multipliers to their probabilities
        height_multipliers: mapping from height multipliers to their probabilities
        min_width_overlap: minimum overlap between the original and the augmented (width)
        min_height_overlap: minimum overlap between the original and the augmented (height)
        """
        super().__init__(
            per_sample=per_sample,
            width_probabilities=width_multipliers,
            height_probabilities=height_multipliers,
            interacting=[True],
        )
        self.interaction_selection: InteractionSelectionPolicy = interaction_selection
        self.min_width_overlap: float = min_width_overlap
        self.min_height_overlap: float = min_height_overlap

    def get_target_interactions(
        self, couple_interactions: Sequence[Interaction]
    ) -> Sequence[int]:
        if self.interaction_selection == InteractionSelectionPolicy.ALL:
            target_interactions = list(range(len(couple_interactions)))
        elif self.interaction_selection == InteractionSelectionPolicy.RANDOM_ONE:
            target_interactions = [random.randint(0, len(couple_interactions) - 1)]
        elif self.interaction_selection == InteractionSelectionPolicy.LARGEST:
            target_interactions = [
                sorted(
                    enumerate(couple_interactions),
                    key=lambda index2interaction: index2interaction[1][
                        "interaction_area"
                    ],
                )[-1][0]
            ]

        elif self.interaction_selection == InteractionSelectionPolicy.SMALLER:
            target_interactions = [
                sorted(
                    enumerate(couple_interactions),
                    key=lambda index2interaction: index2interaction[1][
                        "interaction_area"
                    ],
                )[0][0]
            ]
        else:
            raise NotImplementedError

        return target_interactions

    def augment(
        self,
        target_interaction: int,
        couple_interactions: Sequence[Interaction],
        gene1: str,
        gene2: str,
        dataset: "RNADataset",
    ) -> AugmentResult:
        interaction: Interaction = couple_interactions[target_interaction]

        gene1_info: Mapping[str, Any] = dataset.gene2info[gene1]
        gene2_info: Mapping[str, Any] = dataset.gene2info[gene2]

        # gene1 refers to dim1 of the matrix, so to the width. Symmetrically, gene2 refers to the height (dim 0)
        gene1_length = len(gene1_info["cdna"])
        gene2_length = len(gene2_info["cdna"])

        # First, sample the target width & height from their distributions
        # TODO: limit lengths to not intersect with other interactions?
        target_width: int = _sample_target_dimension_multiplier(
            bins=self.width_bins,
            probabilities=self.width_probabilities,
            max_size=min(gene1_length, self.max_size),
            starting_dim=interaction["w"],
        )
        target_height: int = _sample_target_dimension_multiplier(
            bins=self.height_bins,
            probabilities=self.height_probabilities,
            max_size=min(gene2_length, self.max_size),
            starting_dim=interaction["h"],
        )

        # We can now crop each dimension separately.
        interaction_x1: int = int(interaction["x1"])
        interaction_x2: int = int(interaction_x1 + interaction["w"])
        target_x1 = _compute_target(
            interaction_p1=interaction_x1,
            interaction_p2=interaction_x2,
            min_overlap=self.min_width_overlap,
            max_length=gene1_length,
            interaction_length=interaction["w"],
            target_length=target_width,
        )
        target_x2 = min(target_x1 + target_width, gene1_length)

        interaction_y1: int = int(interaction["y1"])
        interaction_y2: int = int(interaction_y1 + interaction["h"])
        target_y1 = _compute_target(
            interaction_p1=interaction_y1,
            interaction_p2=interaction_y2,
            min_overlap=self.min_height_overlap,
            max_length=gene2_length,
            interaction_length=interaction["h"],
            target_length=target_height,
        )
        target_y2 = min(target_y1 + target_height, gene2_length)

        return dict(
            bbox=BBOX(
                x1=int(target_x1),
                x2=int(target_x2),
                y1=int(target_y1),
                y2=int(target_y2),
            ),
            gene1=gene1,
            gene2=gene2,
            interacting=True,
        )


class EasyNegAugment(AugmentPolicy):
    def __init__(
        self,
        per_sample: Union[float, int],
        width_windows: Mapping[Tuple[int, int], float],
        height_windows: Mapping[Tuple[int, int], float],
    ):
        super().__init__(
            per_sample,
            width_probabilities=width_windows,
            height_probabilities=height_windows,
            interacting=[False],
        )

    def augment(
        self,
        target_interaction: int,
        couple_interactions: Sequence[Interaction],
        gene1: str,
        gene2: str,
        dataset: "RNADataset",
    ) -> AugmentResult:
        interaction: Interaction = couple_interactions[target_interaction]
        assert len(couple_interactions) <= 1 and not interaction["interacting"]

        gene1_info: Mapping[str, Any] = dataset.gene2info[gene1]
        gene2_info: Mapping[str, Any] = dataset.gene2info[gene2]

        # gene1 refers to dim1 of the matrix, so to the width. Symmetrically, gene2 refers to the height (dim 0)
        gene1_length = len(gene1_info["cdna"])
        gene2_length = len(gene2_info["cdna"])

        # First, sample the target width & height from their distributions
        # TODO: limit lengths to not intersect with other interactions?
        target_width: int = _sample_target_dimension_window(
            bins=self.width_bins,
            probabilities=self.width_probabilities,
            max_size=min(gene1_length, self.max_size),
        )
        target_height: int = _sample_target_dimension_window(
            bins=self.height_bins,
            probabilities=self.height_probabilities,
            max_size=min(gene2_length, self.max_size),
        )

        x1: int = int(
            np.random.randint(0, gene1_length - target_width, (1,))
            if target_width != gene1_length
            else 0
        )
        y1: int = int(
            np.random.randint(0, gene2_length - target_height, (1,))
            if target_height != gene2_length
            else 0
        )

        return dict(
            bbox=BBOX(x1=x1, x2=x1 + target_width, y1=y1, y2=y1 + target_height),
            gene1=gene1,
            gene2=gene2,
            interacting=False,
        )


class HardNegAugment(AugmentPolicy):

    _NUM_TRIES: int = (
        150  # 150 tries should be enough given the sparsity of interactions...
    )

    def __init__(
        self,
        per_sample: Union[float, int],
        width_windows: Mapping[Tuple[int, int], float],
        height_windows: Mapping[Tuple[int, int], float],
        # min_width: int = 10,
        # max_height: int = 10,
    ):
        super().__init__(
            per_sample,
            width_probabilities=width_windows,
            height_probabilities=height_windows,
            interacting=[True],
        )

        # self.min_width: int = min_width
        # self.max_height: int = max_height

    def get_target_interactions(
        self, couple_interactions: Sequence[Interaction]
    ) -> Sequence[int]:
        return (0,)

    def augment(
        self,
        target_interaction: int,
        couple_interactions: Sequence[Interaction],
        gene1: str,
        gene2: str,
        dataset: "RNADataset",
    ) -> AugmentResult:
        gene1_info: Mapping[str, Any] = dataset.gene2info[gene1]
        gene2_info: Mapping[str, Any] = dataset.gene2info[gene2]

        # gene1 refers to dim1 of the matrix, so to the width. Symmetrically, gene2 refers to the height (dim 0)
        gene1_length = len(gene1_info["cdna"])
        gene2_length = len(gene2_info["cdna"])

        # First, sample the target width & height from their distributions
        # TODO: limit lengths to not intersect with other interactions?
        target_width: int = _sample_target_dimension_window(
            bins=self.width_bins,
            probabilities=self.width_probabilities,
            max_size=min(gene1_length, self.max_size),
        )
        target_height: int = _sample_target_dimension_window(
            bins=self.height_bins,
            probabilities=self.height_probabilities,
            max_size=min(gene2_length, self.max_size),
        )

        full_matrix = torch.zeros(gene2_length, gene1_length)
        
        for interaction in couple_interactions:
            assert interaction["interacting"]
            interaction_bbox: BBOX = BBOX.from_interaction(interaction)
            full_matrix[
                interaction_bbox.y1 : interaction_bbox.y2,
                interaction_bbox.x1 : interaction_bbox.x2,
            ] = 1

        for _ in range(HardNegAugment._NUM_TRIES):
            x1 = np.random.randint(low=0, high=gene1_length-target_width)
            y1 = np.random.randint(low=0, high=gene2_length-target_height)

            try:
                sample_interaction: torch.Tensor = full_matrix[
                    y1 : y1 + target_height, x1 : x1 + target_width
                ]
                if (
                    sample_interaction.sum() > 0
                ):  # if there's any 1, we are including part of an interaction
                    continue
            except IndexError:
                continue
            
            return dict(
                bbox=BBOX(x1=x1, x2=x1 + target_width, y1=y1, y2=y1 + target_height),
                gene1=gene1,
                gene2=gene2,
                interacting=False,
            )
        else:
            raise RuntimeError(
                f"Couldn't find a non-interacting region with {HardNegAugment._NUM_TRIES} tries"
            )


class RegionSpecNegAugment(AugmentPolicy):
    def __init__(
        self,
        per_sample: Union[float, int],
        width_windows: Mapping[Tuple[int, int], float],
        height_windows: Mapping[Tuple[int, int], float],
        target_rna: str = "random",  # generate neg only for the "first" RNA, or the "second" one or for "both" or "random"
        interaction_selection: InteractionSelectionPolicy = InteractionSelectionPolicy.RANDOM_ONE
        # min_width: int = 10,
        # max_height: int = 10,
    ):
        super().__init__(
            per_sample,
            width_probabilities=width_windows,
            height_probabilities=height_windows,
            interacting=[True],
        )
        self.target_rna: str = target_rna
        self.interaction_selection: InteractionSelectionPolicy = interaction_selection

        # self.min_width: int = min_width
        # self.max_height: int = max_height

    def get_target_interactions(
        self, couple_interactions: Sequence[Interaction]
    ) -> Sequence[int]:
        if self.interaction_selection == InteractionSelectionPolicy.ALL:
            raise RuntimeError("Not compatible with this augment policy")
        elif self.interaction_selection == InteractionSelectionPolicy.RANDOM_ONE:
            target_interactions = [random.randint(0, len(couple_interactions) - 1)]
        elif self.interaction_selection == InteractionSelectionPolicy.LARGEST:
            target_interactions = [
                sorted(
                    enumerate(couple_interactions),
                    key=lambda index2interaction: index2interaction[1][
                        "interaction_area"
                    ],
                )[-1][0]
            ]

        elif self.interaction_selection == InteractionSelectionPolicy.SMALLER:
            target_interactions = [
                sorted(
                    enumerate(couple_interactions),
                    key=lambda index2interaction: index2interaction[1][
                        "interaction_area"
                    ],
                )[0][0]
            ]
        else:
            raise NotImplementedError

        return target_interactions

    def _augment(
        self,
        target_interaction: int,
        couple_interactions: Sequence[Interaction],
        gene1: str,
        gene2: str,
        dataset: "RNADataset",
        swapped: bool,
    ):
        interaction: Interaction = couple_interactions[target_interaction]
        gene1_info = dataset.gene2info[gene1]
        gene1_length = len(gene1_info["cdna"])
        # gene1 refers to dim1 of the matrix, so to the width. Symmetrically, gene2 refers to the height (dim 0)
        x1, x2 = interaction.get_boundaries(gene1)

        target_width: int = _sample_target_dimension_multiplier(
            bins=self.width_bins,
            probabilities=self.width_probabilities,
            max_size=min(gene1_length, self.max_size),
            starting_dim=int(x2 - x1),
        )

        x1 = _compute_target(
            interaction_p1=x1,
            interaction_p2=x2,
            min_overlap=1,
            max_length=gene1_length,
            interaction_length=x2 - x1,
            target_length=target_width,
        )
        x2 = min(x1 + target_width, gene1_length)
        
        # if x2-x1<5:
        #     print(f'{x1=} {x2= } {target_width= } {gene1_length=} {gene1=}')
        #     assert False
        
        # gene1_region_type = interaction['gene1_region']
        gene2_region_type = interaction.get(
            "gene2_region", random.choice(["CDS", "UTR5", "UTR3"])
        )

        negative_genes = []
        couples = []
        for gene1_interaction in gene1_info["interactions"]:
            couples.append(gene1_interaction['couple'])
            if not gene1_interaction["interacting"]:
                negative_genes.append(gene1_interaction["gene1"])
                negative_genes.append(gene1_interaction["gene2"])

        negative_genes = set(negative_genes)
        if gene1 in negative_genes:
            negative_genes.remove(gene1)

        if len(negative_genes) == 0:
            if swapped:
                raise RuntimeError(
                    f"At least one negative interaction should be guaranteed... {gene1, gene2}"
                )

            return self._augment(
                target_interaction=target_interaction,
                couple_interactions=couple_interactions,
                gene1=gene2,
                gene2=gene1,
                dataset=dataset,
                swapped=True,
            )

        # filter on pc and not pc
        negative_genes_pc = [
            negative_gene
            for negative_gene in negative_genes
            if dataset.gene2info[negative_gene]["protein_coding"]
        ]
        negative_genes_npc = [
            negative_gene
            for negative_gene in negative_genes
            if not dataset.gene2info[negative_gene]["protein_coding"]
        ]

        random.shuffle(negative_genes_pc)
        random.shuffle(negative_genes_npc)

        if gene2_region_type == "/":
            # we look for another NC. If there's not one, we end up in UTR3/5
            if len(negative_genes_npc) != 0:
                gene2 = negative_genes_npc[0]
                gene2_info = dataset.gene2info[gene2]
                gene2_length = len(gene2_info["cdna"])
                target_height: int = _sample_target_dimension_window(
                    bins=self.height_bins,
                    probabilities=self.height_probabilities,
                    max_size=min(gene2_length, self.max_size),
                )

                y1: int = int(
                    np.random.randint(0, gene2_length - target_height, (1,))
                    if target_height != gene2_length
                    else 0
                )
                y2: int = min(y1 + target_height, gene2_length)
                
            else:
                region = random.choice(["UTR3", "UTR5"])
                gene2 = negative_genes_pc[0]
                gene2_info = dataset.gene2info[gene2]
                gene2_length = len(gene2_info["cdna"])
                target_height: int = _sample_target_dimension_window(
                    bins=self.height_bins,
                    probabilities=self.height_probabilities,
                    max_size=min(gene2_length, self.max_size),
                )

                # It can be that gene2_info[f"{region}_start"] == gene2_info[f"{region}_end"]
                # because some rna are only UTR3, only UTR5 or only CDS, so I need an if-else statement for these cases.
                # If I fall in the except, I will fall inside a random region of the gene2.
                # (I can fall inside a CDS, but they are few cases).
                if gene2_info[f"{region}_start"] < gene2_info[f"{region}_end"]:
                    fake_y1: int = np.random.randint(gene2_info[f"{region}_start"], 
                                                     gene2_info[f"{region}_end"] - 1)
                    fake_y2:int = fake_y1+1
                    y1: int = _compute_target(
                        interaction_p1=fake_y1,
                        interaction_p2=fake_y2,
                        min_overlap=1,
                        max_length=gene2_length,
                        interaction_length=fake_y2-fake_y1,
                        target_length=target_height,
                    )
                else:
                    y1: int = int(
                        np.random.randint(0, gene2_length - target_height, (1,))
                        if target_height != gene2_length
                        else 0
                    )
                y2: int = min(y1 + target_height, gene2_length)

        elif gene2_region_type in {"CDS", "UTR3", "UTR5"}:
            # we look for another PC. If there's not one, we search randomly
            if len(negative_genes_pc) != 0:
                gene2 = negative_genes_pc[0]
                gene2_info = dataset.gene2info[gene2]
                gene2_length = len(gene2_info["cdna"])
                target_height: int = _sample_target_dimension_window(
                    bins=self.height_bins,
                    probabilities=self.height_probabilities,
                    max_size=min(gene2_length, self.max_size),
                )

                # It can be that gene2_info[f"{region}_start"] == gene2_info[f"{region}_end"]
                # because some rna are only UTR3, only UTR5 or only CDS, so I need an if-else statement for these cases.
                # If I fall in the except, I will fall inside a random region of the gene2.
                # (I can fall inside a CDS, but they are few cases).
                if (
                    gene2_info[f"{gene2_region_type}_start"]
                    < gene2_info[f"{gene2_region_type}_end"]
                ):
                    fake_y1: int = np.random.randint(gene2_info[f"{gene2_region_type}_start"],
                                                     gene2_info[f"{gene2_region_type}_end"] - 1)
                    fake_y2:int = fake_y1+1
                    y1: int = _compute_target(
                        interaction_p1=fake_y1,
                        interaction_p2=fake_y2,
                        min_overlap=1,
                        max_length=gene2_length,
                        interaction_length=fake_y2-fake_y1,
                        target_length=target_height,
                    )

                else:
                    y1: int = int(
                        np.random.randint(0, gene2_length - target_height, (1,))
                        if target_height != gene2_length
                        else 0
                    )

                y2: int = min(y1 + target_height, gene2_length)
                
            else:
                gene2 = negative_genes_npc[0]
                gene2_info = dataset.gene2info[gene2]
                gene2_length = len(gene2_info["cdna"])
                target_height: int = _sample_target_dimension_window(
                    bins=self.height_bins,
                    probabilities=self.height_probabilities,
                    max_size=min(gene2_length, self.max_size),
                )

                y1: int = int(
                    np.random.randint(0, gene2_length - target_height, (1,))
                    if target_height != gene2_length
                    else 0
                )
                y2: int = min(y1 + target_height, gene2_length)
                
        else:
            raise NotImplementedError

        # sample the target height from its distributions
        # TODO: limit lengths to not intersect with other interactions?
        
        if gene1 + '_' + gene2 in couples: #I need the couple_id to be precise
            return dict(
                bbox=BBOX(x1=x1, x2=x2, y1=y1, y2=y2),
                gene1=gene1,
                gene2=gene2,
                interacting=False,
            )

        elif gene2 + '_' + gene1 in couples:
            return dict(
                bbox=BBOX(x1=y1, x2=y2, y1=x1, y2=x2),
                gene1=gene2,
                gene2=gene1,
                interacting=False,
            )
        else:
            raise RuntimeError(
                    f"Which is the couple_id?.. {gene1, gene2}"
                )

            
    def augment(
        self,
        target_interaction: int,
        couple_interactions: Sequence[Interaction],
        gene1: str,
        gene2: str,
        dataset: "RNADataset",
    ) -> AugmentResult:
        assert self.target_rna == "random", f"Not supported {self.target_rna=}"

        genes = [gene1, gene2]
        random.shuffle(genes)
        gene1, gene2 = genes

        return self._augment(
            target_interaction=target_interaction,
            couple_interactions=couple_interactions,
            gene1=gene1,
            gene2=gene2,
            dataset=dataset,
            swapped=False,
        )


def _compute_target(
    interaction_p1: int,
    interaction_p2: int,
    min_overlap: float,
    max_length: int,
    interaction_length: int,
    target_length: int,
):
    # TODO: implement constraint stuff with min_p1, max_p1, min_p2, max_p2 parameters
    min_target_p: int = max(
        0,
        int(interaction_p1 + min_overlap * interaction_length) - target_length,
    )
    max_target_p: int = min(
        max_length - target_length, #max_length - interaction_length
        int(interaction_p2 - min_overlap * interaction_length),
    )

    if min_target_p == max_target_p:
        return min_target_p
    # TODO: what if the drawn length can't be applied here?

    return np.random.randint(min_target_p, max_target_p, (1,))[0]


class RNADataset(Dataset):
    def __init__(
        self,
        gene2info: pd.DataFrame,
        interactions: pd.DataFrame,
        subset_file: Path,
        augment_policies: Collection[AugmentPolicy],
    ):
        self.subset_file = subset_file

        self.gene2info = gene2info
        self.gene2info.rename(
            columns={
                "UTR5": "UTR5_end",
                "CDS": "CDS_end",
                "UTR3": "UTR3_end",
            },
            inplace=True,
        )
    
        self.gene2info["UTR3_start"] = self.gene2info["CDS_end"]
        self.gene2info["CDS_start"] = self.gene2info["UTR5_end"]

        self.gene2info: Mapping[str, Mapping[str, Any]] = {
            item["gene_id"]: item for item in self.gene2info.to_dict(orient="records")
        }

        for gene_info in self.gene2info.values():
            gene_info["interactions"] = []

        interactions['matrix_area'] = interactions.length_1*interactions.length_2
        interactions['interaction_area'] = interactions.w*interactions.h
        interactions.rename(columns={"couples": "couple"}, inplace=True)

        if os.path.isfile(self.subset_file):
            with open(self.subset_file, "rb") as fp:  # Unpickling
                subset = pickle.load(fp)
                interactions = interactions[
                    interactions.couple.isin(subset)
                ].reset_index(drop=True)

        neg_genes = set(interactions[interactions.interacting == False].gene1).union(
            set(interactions[interactions.interacting == False].gene2)
        )

        interactions: Sequence[Dict[str, Any]] = (
            interactions.to_dict(orient="records")
            # * 42  # TODO: remove multiplier, it is here just for benchmark purposes
        )
        interactions = [Interaction(**interaction) for interaction in interactions]
        for interaction, gene_index in itertools.product(
            interactions, ("gene1", "gene2")
        ):
            self.gene2info.setdefault(
                interaction[gene_index],
                {"interactions": [], "gene_id": interaction[gene_index]},
            )["interactions"].append(interaction)

        # Aggregate interactions by RNAs involved
        all_pair_interactions: Sequence[Sequence[AugmentSpec]] = [
            list(values)
            for key, values in groupby(
                sorted(interactions, key=operator.itemgetter("couple")),
                operator.itemgetter("couple"),
            )
        ]
        #print(all_pair_interactions[0]) --> [{'couple': 'ENSG00000000419_ENSG00000249150', 'gene1': 'ENSG00000000419', 'gene2': 'ENSG00000249150', 'interacting': False, ...}]

        self.all_pair_interactions = {
            frozenset(
                (pair_interactions[0]["gene1"], pair_interactions[0]["gene2"])
            ): pair_interactions
            for pair_interactions in all_pair_interactions
        }

        self.augment_specs = [
            augment_spec
            for pair, pair_interactions in self.all_pair_interactions.items()  # frozenset({'ENSG00000132294', 'ENSG00000000003'})
            for augment_policy in augment_policies  # <dataset.data.EasyNegAugment object at 0x7ff00ef45f10>
            # if type(augment_policy).__name__ != "RegionSpecNegAugment"
            if type(augment_policy) != RegionSpecNegAugment
            or any(gene in neg_genes for gene in pair)
            for augment_spec in augment_policy.generate_augment_specs(
                couple_interactions=pair_interactions,
            )
        ]

    def __getitem__(self, item: int) -> Sample:
        """ """
        augment_spec: AugmentSpec = self.augment_specs[item] #{'augment_policy': <dataset.data.EasyPosAugment object at 0x7f96bc340370>, 'couple_interactions': [{'couple': 'ENSG00000000419_ENSG00000280193', 'gene1': 'ENSG00000000419', 'gene2': 'ENSG00000280193', 'interacting': True, ....}
        augment_policy: AugmentPolicy = augment_spec["augment_policy"]
        couple_interactions = augment_spec["couple_interactions"]

        gene1: str = couple_interactions[0]["gene1"]
        gene2: str = couple_interactions[0]["gene2"]

        policy = augment_policy.name

        augment_result: AugmentResult = augment_policy.augment(
            target_interaction=augment_spec["target_interaction"],
            couple_interactions=couple_interactions,
            gene1=gene1,
            gene2=gene2,
            dataset=self,
        )

        interaction = couple_interactions[augment_spec["target_interaction"]]
        interaction_bbox: BBOX = BBOX.from_xyhw(
            x=interaction["x1"],
            y=interaction["y1"],
            width=interaction["w"],
            height=interaction["h"],
        )

        gene1 = augment_result["gene1"]
        gene2 = augment_result["gene2"]
        all_couple_interactions = self.all_pair_interactions[frozenset((gene1, gene2))]
        
        if (np.random.rand()<0.5): #in regionneg the first gene is always the fixed one
            bbox = augment_result["bbox"]
            s_bbox = interaction_bbox
            return Sample(
                gene1=gene2,
                gene2=gene1,
                couple_id = gene1 + '_' + gene2,
                bbox=BBOX(bbox.y1, bbox.y2, bbox.x1, bbox.x2),
                policy=policy,
                # interaction=augment_spec["target_interaction"],
                seed_interaction_bbox=BBOX(s_bbox.y1, s_bbox.y2, s_bbox.x1, s_bbox.x2),
                interacting=augment_result["interacting"],
                all_couple_interactions=all_couple_interactions,
                gene1_info=self.gene2info[gene2],
                gene2_info=self.gene2info[gene1],
            )
        else:
            return Sample(
                gene1=gene1,
                gene2=gene2,
                couple_id = gene1 + '_' + gene2,
                bbox=augment_result["bbox"],
                policy=policy,
                # interaction=augment_spec["target_interaction"],
                seed_interaction_bbox=interaction_bbox,
                interacting=augment_result["interacting"],
                all_couple_interactions=all_couple_interactions,
                gene1_info=self.gene2info[gene1],
                gene2_info=self.gene2info[gene2],
            )
            

    def __len__(self):
        return len(self.augment_specs)


def _sample_target_dimension_window(bins, probabilities, max_size: int) -> int:
    target_bin_index: torch.Tensor = torch.multinomial(
        torch.as_tensor(probabilities),
        num_samples=1,
        replacement=True,
    )

    target_low, target_high = bins[target_bin_index]
    target: int = torch.randint(target_low, target_high, (1,)).item()

    return min(target, max_size)


def _sample_target_dimension_multiplier(
    bins, probabilities, starting_dim: int, max_size: int
) -> int:
    target_bin_index: torch.Tensor = torch.multinomial(
        torch.as_tensor(probabilities),
        num_samples=1,
        replacement=True,
    )
    target_multiplier = bins[target_bin_index]
    target: int = int(starting_dim * target_multiplier)

    return min(max(target, starting_dim), max_size)



def plot_sample(sample: Sample, plot_interaction_negatives=False):
    sample_bbox: BBOX = sample.bbox
    real_bbox: BBOX = sample.seed_interaction_bbox
    
    cdna1 = sample.gene1_info["cdna"]
    cdna2 = sample.gene2_info["cdna"]
    width = len(cdna1)
    height = len(cdna2)
    print(width, height)
    # Create figure and axes
    fig, ax = plt.subplots()
    ratio = width/height
    print(ratio)
    fig.set_figwidth(7*ratio)
    fig.set_figheight(7)
    ax.plot([0, width],[0, height], 'ro', color = 'white')
    rect = Rectangle((sample_bbox.x1, sample_bbox.y1), 
                     sample_bbox.x2-sample_bbox.x1, sample_bbox.y2-sample_bbox.y1,
                             linewidth=2, edgecolor='darkblue', fill=False)
    ax.add_patch(rect) # Add the patch to the Axes
    if (sample.interacting | plot_interaction_negatives):
        rect = Rectangle((real_bbox.x1, real_bbox.y1), 
                         real_bbox.x2-real_bbox.x1, real_bbox.y2-real_bbox.y1, 
                         linewidth=2, edgecolor='red', fill=True, facecolor='red',)
        ax.add_patch(rect) # Add the patch to the Axes
    return fig

class FindSplits:
    def __init__(
        self,
        max_size: int = MAX_RNA_SIZE,
    ):
        self.max_size = max_size
        
    def get_split_coords(
        self, length: int, step_size: int,
    ) -> Sequence[Tuple[int, int]]:
        length, max_size = length -1, self.max_size-1 #python indexes
        coords = []
        i = 0
        increase = 0
        stop = False
        while stop == False:
            increase = min(i+max_size, length)
            coords.append((i, increase))
            i+=step_size
            stop = (increase == length)
        # the last one is penalized. Suppose length = 1100, step_size=500,
        # the output is [(0, 511), (500, 1011), (1000, 1099)] 
        # (the last is only 100 nucleotides)
        # I prefer to take the maximum length at the last step. 
        coords.pop()
        coords.append((increase-min(max_size, length), increase))
        return coords
    
class RNADatasetInference(Dataset):
    def __init__(
        self,
        gene_info_path: Path,
        interactions_path: Path,
        step_size: int
    ):
        self.gene_info_path: Path = gene_info_path
        self.interactions_path: Path = interactions_path
        self.step_size = step_size
        
        d = {}
        for i, fasta in enumerate(SeqIO.parse(open(gene_info_path),'fasta')):
            name = str(fasta.description)
            seq = str(fasta.seq)
            d[i] = {'gene_id': name, 
                   'cdna':seq}
            
        self.gene2info: pd.DataFrame = pd.DataFrame.from_dict(d, orient = 'index')
        
        self.gene2info['length'] = self.gene2info.cdna.str.len()
        
        fs = FindSplits(max_size = MAX_RNA_SIZE)
        self.gene2info['coords'] = self.gene2info.length.apply(lambda x: fs.get_split_coords(length = x, step_size=self.step_size))
        
        self.gene2info: Mapping[str, Mapping[str, Any]] = {
            item["gene_id"]: item for item in
            self.gene2info.to_dict(orient="records")
        }
        
        for gene in self.gene2info.keys():
            c1, c2 = zip(*self.gene2info[gene]['coords'])
            df_gene = pd.DataFrame([c1, c2]).T.rename({0:'c1', 1:'c2'}, axis = 1)
            self.gene2info[gene]['df'] = df_gene
        
        self.interactions: pd.DataFrame = pd.read_csv(interactions_path, 
            header = None).rename({0:'pairs'}, axis = 1)
        
        new = self.interactions.pairs.str.split('_', expand = True)
        self.interactions['gene1'],  self.interactions['gene2'] = new[0], new[1]
        dfs = []
        for _, row in self.interactions.iterrows():
            df1 = self.gene2info[row.gene1]['df']
            df1['pairs'] = row.pairs
            df1['gene1'] = row.gene1
            df2 = self.gene2info[row.gene2]['df']
            df2['pairs'] = row.pairs
            df1['gene2'] = row.gene2
            df = df1.merge(df2, on = 'pairs').rename({'c1_x':'x1', 
                                                     'c2_x':'x2',
                                                     'c1_y':'y1',
                                                     'c2_y':'y2'
                                                     }, axis = 1)
            dfs.append(df)

        self.interactions = pd.concat(dfs, axis = 0).reset_index(drop = True)
        del dfs
        
        
    def __getitem__(self, item: int) -> Sample:
        row = self.interactions.loc[item]
        return Sample(
            gene1=row.gene1,
            gene2=row.gene2,
            couple_id = row.gene1 + '_' + row.gene2,
            bbox=BBOX(row.x1, row.x2, row.y1, row.y2),
            policy='inference',
            seed_interaction_bbox=BBOX(np.nan,np.nan,np.nan,np.nan), #fake
            interacting=np.nan, #Fake
            all_couple_interactions=[], #fake
            gene1_info=self.gene2info[row.gene1],
            gene2_info=self.gene2info[row.gene2],
        )
    def __len__(self):
        return self.interactions.shape[0]
