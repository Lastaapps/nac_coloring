from itertools import product
from typing import Iterator, Tuple


def lazy_product[T, R](iter1: Iterator[T], iter2: Iterator[R]) -> Iterator[Tuple[T, R]]:
    """
    Alternative for itertools.product that exhausts both the iterators in parallel.
    In the standard implementation, first the first iterator is exhausted
    and then second element of the other iterator is read.
    Here both the iterators switch roles in who takes the next element.

    This function is slightly slower and requires more memory as it's not
    written in C, on the other hand it can provide some combinations
    sooner and some early termining algorithms may find it beneficial.
    Also, the itertools.product is blocking (it waits for the result),
    this is not.

    Returns
    -------
        The same as itertools.product, but in different order.

    Examples
    --------
    >>> a, b = [1, 2, 3], ["a", "b", "c"]
    >>> list(itertools.product(a, b))
    [(1, 'a'), (1, 'b'), (1, 'c'),
     (2, 'a'), (2, 'b'), (2, 'c'),
     (3, 'a'), (3, 'b'), (3, 'c')]

    >>> list(lazy_product(a, b)
    [(1, 'a'), (2, 'a'), (1, 'b'),
     (2, 'b'), (3, 'a'), (3, 'b'),
     (1, 'c'), (2, 'c'), (3, 'c')]
    """
    cache1 = []
    cache2 = []

    iter1 = iter(iter1)
    iter2 = iter(iter2)

    # product is not used in the implementation for it's blocking nature
    while True:
        res = next(iter1, None)
        if res is not None:
            cache1.append(res)
            yield from ((res, c) for c in cache2)
        else:
            break

        res = next(iter2, None)
        if res is not None:
            cache2.append(res)
            yield from ((c, res) for c in cache1)
        else:
            break

    # one of those is already empty, so at most one of these lines will run
    yield from ((c1, c2) for c1 in iter1 for c2 in cache2)
    yield from ((c1, c2) for c2 in iter2 for c1 in cache1)
