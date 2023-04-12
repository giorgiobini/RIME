# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detrnarna import build

from .mlp import build as build_naive_mlp


def build_model(args):
    return build(args)

def build_mlp(args):
    return build_naive_mlp(args)