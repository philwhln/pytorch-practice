from typing import Iterable


def enumerate_with_print(xs: Iterable) -> Iterable:
    for x in xs:
        print(f'x = {x}')
        yield x
    print('xs done')


it = map(lambda x: x ** 2, [1, 3, 4, 5, 6])
for i in enumerate_with_print(it):
    print(f'i = {i}')
print('it done')
