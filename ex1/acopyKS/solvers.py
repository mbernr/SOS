# -*- coding: utf-8 -*-
import sys
import functools
import collections
import numpy as np

from . import utils


@functools.total_ordering
class Solution:
    """Binary vector (items of the knapsack that are picked)

    :param instance: an instance
    :type instance: :class:`utils.KnapsackInstance`
    :param start: starting item (by default 0)
    :param ant: ant responsible
    :type ant: :class:`~acopy.ant.Ant`
    """

    def __init__(self, instance, ant=None):
        self.instance = instance
        self.ant = ant
        self.value = 0
        self.weight = 0
        self.picked_items = np.zeros(instance.number_items, dtype=int)
        self.items_to_pick = [i for i in range(0, instance.number_items)]

    def __iter__(self):
        return iter(self.path)

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __contains__(self, item):
        return self.picked_items[item]==1

    def __repr__(self):
        easy_id = self.get_easy_id(sep=',', monospace=False)
        return '{}\t{}'.format(self.value, easy_id)

    def __hash__(self):
        return hash(self.get_id())

    def add_item(self, item):
        """Record an item as picked or left

        :param item: the item wich is getting picked / left
        """
        self.items_to_pick.remove(item)
        self.picked_items[item]=1
        self.value+=self.instance.values_items[item]
        self.weight+=self.instance.weights_items[item]
        self.items_to_pick = [item for item in self.items_to_pick if self.instance.weights_items[item] < (self.instance.capacity-self.weight)]

    def get_easy_id(self, sep=' ', monospace=True):
        return str((self.value, self.weight))

    def trace(self, q, rho=0):
        """Deposit pheromone on the items.

        Note that by default no pheromone evaporates.

        :param float q: the amount of pheromone
        :param float rho: the percentage of pheromone to evaporate
        """
        amount = q * self.value
        for item in range(0,self.instance.number_items):
            self.instance.pheromone[item] += amount
            self.instance.pheromone[item] *= 1 - rho
            if self.instance.pheromone[item] < 0.0:
                self.instance.pheromone[item] = 0.0


class State:
    """Solver state.

    This class tracks the state of a solution in progress and is passed to each
    plugin hook. Specically it conatins:

    ===================== ======================================
    Attribute             Description
    ===================== ======================================
    ``instance`           instance being solved
    ``colony``            colony that generated the ants
    ``ants``              ants being used to solve the graph
    ``limit``             maximum number of iterations
    ``gen_size``          number of ants being used
    ``solutions``         solutions found this iteration
    ``best``              best solution found this iteration
    ``is_new_record``     whether the best is a new record
    ``record``            best solution found so far
    ``previous_record``   previously best solution
    ===================== ======================================

    :param graph: a graph
    :type graph: :class:`networkx.Graph`
    :param list ants: the ants being used
    :param int limit: maximum number of iterations
    :param int gen_size: number of ants to use
    :param colony: source colony for the ants
    :type colony: :class:`~acopy.ant.Colony`
    """

    def __init__(self, instance, ants, limit, gen_size, colony):
        self.instance = instance
        self.ants = ants
        self.limit = limit
        self.gen_size = gen_size
        self.colony = colony
        self.solutions = None
        self.record = None
        self.previous_record = None
        self.is_new_record = False
        self._best = None

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, best):
        self.is_new_record = self.record is None or best > self.record
        if self.is_new_record:
            self.previous_record = self.record
            self.record = best
        self._best = best


class Solver:
    """ACO solver.

    Solvers control the parameters related to phomone deposit and evaporation.
    If top is not specified, it defaults to the number of ants used to solve a
    graph.

    :param float rho: percentage of pheromone that evaporates each iteration
    :param float q: amount of pheromone each ant can deposit
    :param int top: number of ants that deposit pheromone
    :param list plugins: zero or more solver plugins
    """

    def __init__(self, rho=.03, q=1, top=None, plugins=None):
        self.rho = rho
        self.q = q
        self.top = top
        self.plugins = collections.OrderedDict()
        if plugins:
            self.add_plugins(*plugins)

    def __repr__(self):
        return (f'{self.__class__.__name__}(rho={self.rho}, q={self.q}, '
                f'top={self.top})')

    def solve(self, *args, **kwargs):
        """Find and return the best solution.

        Accepts exactly the same paramters as the :func:`~optimize` method.

        :return: best solution found
        :rtype: :class:`~Solution`
        """
        best = None
        for solution in self.optimize(*args, **kwargs):
            best = solution
        return best

    def optimize(self, instance, colony, gen_size=None, limit=None):
        """Find and return increasingly better solutions.

        :param graph: instance to solve
        :type graph: :class:`utils.KnapsackInstance`
        :param colony: colony from which to source each :class:`~acopy.ant.Ant`
        :type colony: :class:`~acopy.ant.Colony`
        :param int gen_size: number of :class:`~acopy.ant.Ant` s to use
                             (default is one per graph node)
        :param int limit: maximum number of iterations to perform (default is
                          unlimited so it will run forever)
        :return: better solutions as they are found
        :rtype: iter
        """
        # initialize the colony of ants and the graph
        gen_size = gen_size or len(graph.nodes)
        ants = colony.get_ants(gen_size)

        state = State(instance=instance, ants=ants, limit=limit, gen_size=gen_size,
                      colony=colony)

        # call start hook for all plugins
        self._call_plugins('start', state=state)

        # find solutions and update the graph pheromone accordingly
        for __ in utils.looper(limit):
            solutions = self.find_solutions(state.instance, state.ants)

            # we want to ensure the ants are sorted with the solutions, but
            # since ants aren't directly comparable, so we interject a list of
            # unique numbers that satifies any two solutions that are equal
            data = list(zip(solutions, range(len(state.ants)), state.ants))
            data.sort(reverse=True)
            solutions, __, ants = zip(*data)

            state.solutions = solutions
            state.ants = ants
            self.global_update(state)

            # yield increasingly better solutions
            state.best = state.solutions[0]
            if state.is_new_record:
                yield state.record

            # call iteration hook for all plugins
            if self._call_plugins('iteration', state=state):
                break

        # call finish hook for all plugins
        self._call_plugins('finish', state=state)

    def find_solutions(self, instance, ants):
        """Return the solutions found for the given ants.

        :param instance: an instance
        :type instance: :class:`utils.KnapsackInstance`
        :param list ants: the ants to use
        :return: one solution per ant
        :rtype: list
        """
        return [ant.tour(instance) for ant in ants]

    def global_update(self, state):
        """Perform a global pheromone update.

        :param state: solver state
        :type state: :class:`~State`
        """
        for item in range(0, state.instance.number_items):
            amount = 0
            if self.top:
                solutions = state.solutions[:self.top]
            else:
                solutions = state.solutions
            for solution in solutions:
                if solution.picked_items[item]==1:
                    amount += self.q * solution.value / solution.instance.ub
            p = state.instance.pheromones[item]
            state.instance.pheromones[item] = (1 - self.rho) * p + amount

    def add_plugin(self, plugin):
        """Add a single solver plugin.

        If plugins have the same name, only the last one added is kept.

        :param plugin: the plugin to add
        :type plugin: :class:`acopy.plugins.SolverPlugin`
        """
        self.add_plugins(plugin)

    def add_plugins(self, *plugins):
        """Add one or more solver plugins."""
        for plugin in plugins:
            plugin.initialize(self)
            self.plugins[plugin.__class__.__qualname__] = plugin

    def get_plugins(self):
        """Return the added plugins.

        :return: plugins previously added
        :rtype: list
        """
        return self.plugins.values()

    def _call_plugins(self, hook, **kwargs):
        should_stop = False
        for plugin in self.get_plugins():
            try:
                plugin(hook, **kwargs)
            except StopIteration:
                should_stop = True
        return should_stop


class SolverPlugin:
    """Solver plugin.

    Solver plugins can be added to any solver to customize its behavior.
    Plugins are initilized once when added, once before the first solver
    iteration, once after each solver iteration has completed, and once after
    all iterations have completed.

    Implenting each hook is optional.
    """

    #: unique name
    name = 'plugin'

    def __init__(self, **kwargs):
        self._params = kwargs

    def __repr__(self):
        params = ', '.join(f'{k}={v}'for k, v in self._params.items())
        return f'<{self.__class__.__qualname__}({params})>'

    def __call__(self, hook, **kwargs):
        return getattr(self, f'on_{hook}')(**kwargs)

    def initialize(self, solver):
        """Perform actions when being added to a solver.

        Though technically not required, this method should be probably be
        idempotent since the same plugin could be added to the same solver
        multiple times (perhaps even by mistake).

        :param solver: the solver to which the plugin is being added
        :type solver: :class:`acopy.solvers.Solver`
        """
        self.solver = solver

    def on_start(self, state):
        """Perform actions before the first iteration.

        :param state: solver state
        :type state: :class:`acopy.solvers.State`
        """
        pass

    def on_iteration(self, state):
        """Perform actions after each iteration.

        :param state: solver state
        :type state: :class:`acopy.solvers.State`
        """
        pass

    def on_finish(self, state):
        """Perform actions once all iterations have completed.

        :param state: solver state
        :type state: :class:`acopy.solvers.State`
        """
        pass
