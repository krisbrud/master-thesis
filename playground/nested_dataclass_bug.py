from dataclasses import dataclass
from copy import deepcopy

@dataclass
class Inner:
    foo: int = 1

@dataclass
class Outer:
    inner: Inner = Inner()

outer1 = deepcopy(Outer())
outer1.inner.foo = 2

outer2 = Outer()
print(outer2.inner.foo) 