from scstmatch.data import Dataset


class InputProxy:
    def __init__(self, inputs: dict):
        self.inputs = inputs

    def __getattr__(self, key):
        if key in self.inputs:
            return self.inputs[key]
        raise AttributeError(f"Unknown input {key}.")

    def __dir__(self):
        return self.inputs.keys()


class OptionsProxy:
    def __init__(self, defaults: dict):
        self._defaults = defaults
        self._options = dict()
        for key, value in defaults.items():
            self._options.setdefault(key, value)

    def update(self, options):
        self._options.update(options)

    def override(self, overrides):
        overridden = OptionsProxy(self._defaults)
        overridden.update(self._options)
        overridden.update(overrides)
        return overridden

    def __ensure_key(self, key):
        if key not in self._defaults:
            raise AttributeError(f"Unknown option {key}.")

    def __getattr__(self, key):
        self.__ensure_key(key)
        return self._options[key]

    def __setattr__(self, key, value):
        if key in ["_defaults", "_options"]:
            return super().__setattr__(key, value)
        self.__ensure_key(key)
        self._options[key] = value

    def __dir__(self):
        return self.defaults.keys()


class Generator:
    def __init__(self, inputs=None, defaults=None, **options):
        self.inputs = InputProxy(inputs or dict())
        self._update_inputs()
        self.options = OptionsProxy(defaults or dict())
        self.reconfigure(**options)

    def _update_inputs(self):
        """
        Called when the inputs of this generator are set (once during initialization).
        Can be used to precompute values used later during generation.
        """
        pass

    def reconfigure(self, **options):
        self.options.update(options)

    def _generate(self) -> Dataset:
        pass

    def generate(self, **overrides) -> Dataset:
        old_options = self.options
        self.options = old_options.override(overrides)
        data = self._generate()
        self.options = old_options
        return data
