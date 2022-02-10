class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules (dict of name x :class:`Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        training (bool): Whether the module is in training mode or evaluation mode

    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self):
        "Return the direct child modules of this module."
        return self.__dict__["_modules"].values()

    def get_direct_parameters(self):
        return self.__dict__["_parameters"]

    def get_direct_modules(self):
        return self.__dict__["_modules"]

    def train(self):
        "Set the mode of this module and all descendent modules to `train`."
        queue = [self]
        while queue:
            current_modules = queue.pop(0)
            current_modules.training = True
            for neighbour in current_modules.modules():
                queue.append(neighbour)

    def eval(self):
        "Set the mode of this module and all descendent modules to `eval`."
        queue = [self]
        while queue:
            current_module = queue.pop(0)
            current_module.training = False
            for direct_module in current_module.modules():
                queue.append(direct_module)

    def named_parameters(self):
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            list of pairs: Contains the name and :class:`Parameter` of each ancestor parameter.
        """
        parameters = []
        queue = [("", self)]
        while queue:
            current_module = queue.pop(0)
            current_module_name = current_module[0]

            params = current_module[1].get_direct_parameters()
            for parameter_name in params.keys():
                absolute_parameter_name = parameter_name
                if current_module_name != "":
                    absolute_parameter_name = (
                        current_module_name + "." + absolute_parameter_name
                    )
                parameters.append((absolute_parameter_name, params[parameter_name]))

            direct_modules = current_module[1].get_direct_modules()
            for direct_module_name in direct_modules.keys():
                absolute_direct_module_name = direct_module_name
                if current_module_name != "":
                    absolute_direct_module_name = (
                        current_module_name + "." + absolute_direct_module_name
                    )
                queue.append(
                    (absolute_direct_module_name, direct_modules[direct_module_name])
                )

        return parameters

    def parameters(self):
        "Enumerate over all the parameters of this module and its descendents."
        named_params = self.named_parameters()
        params = []
        for param in named_params:
            params.append(param[1])
        return params

    def add_parameter(self, k, v):
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k (str): Local name of the parameter.
            v (value): Value for the parameter.

        Returns:
            Parameter: Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        assert False, "Not Implemented"

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x=None, name=None):
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x):
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)
