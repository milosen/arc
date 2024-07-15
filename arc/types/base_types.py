import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy
from os import PathLike
from typing import Dict, Any, TypeVar, Union

from arc.controls.common import *

RegisterType = TypeVar("RegisterType", bound="Register")


class Element(ABC):
    id: str
    info: Dict[str, Any]

    def __getitem__(self, item):
        return self.get_elements()[item]

    def __iter__(self):
        return iter(self.get_elements())

    def __str__(self):
        return self.id

    @abstractmethod
    def get_elements(self):
        pass


class Register(OrderedDict):
    MAX_PRINT_ELEMENTS = 10
    INFO_KEY = "_info"

    def __init__(self, other=(), /, **kwargs):
        if self.INFO_KEY in kwargs:
            self.info = kwargs[self.INFO_KEY]
            del kwargs[self.INFO_KEY]
        else:
            self.info = {}

        super().__init__(other, **kwargs)

    def __contains__(self, item: Union[str, Element]):
        if isinstance(item, str):
            return item in self.keys()
        elif isinstance(item, Element):
            return item.id in self.keys()
        else:
            raise ValueError("item type unknown")

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value: Dict):
        self._info = value

    def __iter__(self):
        return iter(self.values())

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return list(self.values())[item]

        return super().__getitem__(item)

    def __str__(self):
        li = list(self.keys())
        n_elements = len(li)

        s = "|".join(li[:self.MAX_PRINT_ELEMENTS])

        if n_elements > self.MAX_PRINT_ELEMENTS:
            s += "|..."

            s += f" ({n_elements} elements total)"

        return s

    def append(self, obj: Element):
        self[str(obj)] = obj

    def get_subset(self, size: int) -> RegisterType:
        """Create a new Register as a random subset of this one"""
        if size >= len(self):
            return self

        keys = set()

        for _ in range(size):
            keys.add(random.choice(list(self.keys() - keys)))

        return self.new_from_dict({key: self[key] for key in keys})

    def get_self_with_info_key(self):
        d = copy(self)
        d.update({self.INFO_KEY: self.info})
        return d

    def to_json(self):
        return json.dumps(self.get_self_with_info_key(), default=lambda o: o.model_dump(),
                          sort_keys=False, ensure_ascii=False)

    def save(self, path: Union[str, PathLike] = None):
        if path is None:
            path = f"{self[0].__class__.__name__.lower()}s.json"

        if isinstance(path, str) and not path.endswith(".json"):
            path = path + ".json"

        with open(path, "w", encoding='utf-8') as file:
            json.dump(self.get_self_with_info_key(), file,
                      default=lambda o: o.model_dump(), sort_keys=False, ensure_ascii=False)

    def intersection(
            self,
            other: RegisterType
    ) -> RegisterType:
        """
        Select Elements that are in both registers and merge the data
        :param other:
        :return:
        """

        def merge_infos(element_1, element_2):
            element_new = copy(element_1)
            element_new.info.update(element_2.info)
            return element_new

        new_register = self.new_from_dict({
            key: merge_infos(element, other[key]) for key, element in self.items() if key in other
        })

        new_register.info.update(other.info)

        return new_register

    def new_from_dict(self, dictionary: dict) -> RegisterType:
        new_info = copy(self.info)
        return Register(dictionary, _info=new_info)
    
    def empty_like(self):
        return Register({}, _info=copy(self.info))

    def flatten(self) -> RegisterType:
        reg = self.empty_like()
        for element in self:
            for sub_element in element.get_elements():
                reg.append(sub_element)
        return reg
    
    def filter(self, func, *args, **kwargs):
        reg = self.empty_like()
        for element in self:
            if func(element, *args, **kwargs):
                reg.append(element)
        return reg
