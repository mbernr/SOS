# -*- coding: utf-8 -*-
import sys
import itertools
import bisect
import random

from .utils import positive
from .solvers import Solution


class Ant:
    """An ant.

    Ants explore a graph, using alpha and beta to guide their decision making
    process when choosing which edge to travel next.

    :param float alpha: how much pheromone matters
    :param float beta: how much distance matters
    """

    def __init__(self, alpha=1, beta=3):
        self.alpha = alpha
        self.beta = beta

    @property
    def alpha(self):
        """How much pheromone matters. Always kept greater than zero."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = positive(value)

    @property
    def beta(self):
        """How much distance matters. Always kept greater than zero."""
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = positive(value)

    def __repr__(self):
        return f'Ant(alpha={self.alpha}, beta={self.beta})'

    def tour(self, instance):
        """Find a solution to the given instance.

        :param instance: the instance to solve
        :type instance: :class:`utils.KnapsackInstance`
        :return: one solution
        :rtype: :class:`~acopy.solvers.Solution`
        """
        solution = self.initialize_solution(instance)
        items_to_pick = solution.items_to_pick
        while items_to_pick and solution.weight < instance.capacity :
            item = self.choose_item(instance, items_to_pick)
            solution.add_item(item)
            items_to_pick = solution.items_to_pick
        return solution

    def initialize_solution(self, instance):
        return Solution(instance, ant=self)

    def choose_item(self, instance, candidates):
        if len(candidates) == 1:
            return candidates[0]
        scores = self.get_scores(instance, candidates)
        return self.choose(candidates, scores)

    def get_scores(self, instance, candidates):
        scores = []
        for item in candidates:
            score = self.score_item(instance, item)
            scores.append(score)
        return scores

    def choose(self, choices, scores):
        """Return one of the choices.

        Note that ``scores[i]`` corresponds to ``choices[i]``.

        :param list choices: the unvisited nodes
        :param list scores: the scores for the given choices
        :return: one of the choices
        """
        total = sum(scores)
        cumdist = list(itertools.accumulate(scores)) + [total]
        index = bisect.bisect(cumdist, random.random() * total)
        return choices[min(index, len(choices) - 1)]

    def score_item(self, instance, item):
        """Return the score for the given item.

        :return: score
        :rtype: float
        """
        weight = instance.weights_items[item]
        value = instance.values_items[item]
        if weight == 0:
            return sys.float_info.max
        pre =  value / weight
        post = instance.pheromones[item]
        return (0.1+post) ** self.alpha * (0.1+pre) ** self.beta


class Colony:
    """Colony of ants.

    Effectively this is a source of :class:`~acopy.ant.Ant` for a
    :class:`~acopy.solvers.Solver`.

    :param float alpha: relative factor for edge weight
    :param float beta: relative factor for edge pheromone
    """

    def __init__(self, alpha=1, beta=3):
        self.alpha = alpha
        self.beta = beta

    def __repr__(self):
        return (f'{self.__class__.__name__}(alpha={self.alpha}, '
                f'beta={self.beta})')

    def get_ants(self, count):
        """Return the requested number of :class:`~acopy.ant.Ant` s.

        :param int count: number of ants to return
        :rtype: list
        """
        return [Ant(**vars(self)) for __ in range(count)]
