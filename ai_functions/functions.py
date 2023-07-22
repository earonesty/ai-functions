import asyncio
import inspect
import json
from typing import get_origin, Any, Annotated, get_args, Callable, Iterable, Union
import logging

log = logging.getLogger(__name__)

JSON_TYPE_MAP = {
    str: "string",
    int: "number",
    bool: "boolean",
    float: "number",
    list: "array"
}


def convert_response(ret):
    # convert content to be a string, for sure
    try:
        content = ret if isinstance(ret, str) else json.dumps(ret)
    except TypeError:
        content = str(ret)
    return content


class AIFunctions:
    def __init__(self, funcs=[], *, loop=None, convert_output=convert_response):
        self.map = dict({f.__name__: f for f in funcs})
        self.loop = loop
        self.validate()
        self.convert_output = convert_output

    def validate(self):
        """Check that all functions have appropriate annotations."""
        get_openai_dict(self.map.values())

    def openai_dict(self, funcs: Iterable[Union[str, Callable]] = []):
        """Make an openai compatible function dict"""
        if funcs:
            names = [f if isinstance(f, str) else f.__name__ for f in funcs]
            funcs = [v for k, v in self.map.items() if k in names]
        else:
            funcs = self.map.values()
        return get_openai_dict(funcs)

    def execute(self, name: str, args: Union[str, dict], *, loop=None, **kws):
        """Run a named function"""
        return execute_function([self.map[name]], name, args, loop=loop or self.loop,
                                convert_output=self.convert_output, **kws)

    def openai_execute(self, call: dict, *, loop=None, **kws):
        """Run an openai style function_call"""
        name, args = call["name"], call["arguments"]
        return execute_function([self.map[name]], name, args, loop=loop or self.loop,
                                convert_output=self.convert_output, **kws)

    async def async_execute(self, name, args, **kws):
        """Async run a named function"""
        return async_execute_function([self.map[name]], name, args, convert_output=self.convert_output, **kws)

    async def async_openai_execute(self, call, **kws):
        """Async run an openai style function"""
        name, args = call["name"], call["arguments"]
        return async_execute_function([self.map[name]], name, args, convert_output=self.convert_output, **kws)

    def __contains__(self, func):
        name = func if isinstance(func, str) else  func.__name__
        return name in self.map

    def description(self, name):
        """Get description of a function"""
        return inspect.getdoc(self.map[name])

    def add(self, func):
        get_openai_dict([func])
        self.map[func.__name__] = func

    def discard(self, func):
        name = func if isinstance(func, str) else  func.__name__
        self.map.pop(name, None)

    def __iter__(self):
        return iter(self.map.values())

def get_openai_args(sig):
    annotated_args: dict[str, dict[str, str]] = {}
    for param_index, (param_name, param) in enumerate(sig.parameters.items()):
        if param_name in ("self", "cls") and param_index == 0:
            continue
        if param.kind == param.POSITIONAL_ONLY:
            return None
        if param.annotation == inspect.Parameter.empty or param.annotation == Any:
            return None
        if get_origin(param.annotation) is not Annotated:
            return None
        base_type = get_args(param.annotation)[0]
        param_type = JSON_TYPE_MAP[base_type]
        param_description = param.annotation.__metadata__[0]
        if param_description is None:
            continue
        annotated_args[param_name] = {"type": param_type, "description": param_description}
    return annotated_args


def get_openai_dict(funcs: Iterable[Callable] = None):
    ret = []
    for func in funcs:
        name = func.__name__
        docs = inspect.getdoc(func)
        sig = inspect.signature(func)
        args = get_openai_args(sig)

        if args is None or docs is None:
            raise ValueError(f"Invalid function {name}, must have annotated docs and arguments")

        ret.append(dict(
            name=name,
            description=docs,
            parameters=dict(
                type="object",
                properties=dict(
                    **args
                ),
            )
        ))

    return ret


def execute_function(funcs: Iterable[Callable], name: str, arguments: Union[str, dict], *, loop=None,
                     convert_output=convert_response, **kws):
    args, func = prepare_function(arguments, funcs, name)
    ret = func(**{**args, **kws})
    while inspect.iscoroutine(ret):
        if not loop:
            raise ValueError(f"Function {name} cannot be run, because we need a loop for async functions")
        ret = asyncio.run_coroutine_threadsafe(ret, loop=loop).result()
    return convert_output(ret) if convert_output else ret


async def async_execute_function(funcs: Iterable[Callable], name: str, arguments: str, convert_output=convert_response, **kws):
    args, func = prepare_function(arguments, funcs, name, **kws)
    ret = await func(**{**args, **kws})
    return convert_output(ret) if convert_output else ret


def prepare_function(arguments: Union[str, dict], funcs: Iterable[Callable], name: str, auto_cast=True):
    func = {f.__name__: f for f in funcs}.get(name)

    if not func:
        raise KeyError(f"Unknown function {name}")

    if isinstance(arguments, str):
        try:
            # we assume json in arguments
            args = json.loads(arguments)
        except Exception as e:
            raise ValueError(f"Could not load arguments for function {name}, invalid JSON: {repr(e)}")
    else:
        args = arguments

    if auto_cast:
        sig = inspect.signature(func)
        for param_index, (param_name, param) in enumerate(sig.parameters.items()):
            if param_name in args:
                arg = args[param_name]
                base_type = get_args(param.annotation)[0]
                if not isinstance(arg, base_type):
                    try:
                        args[param_name] = base_type(arg)
                    except Exception as e:
                        raise ValueError(f"Cannot convert argument {param_name} to type {base_type}: {repr(e)}")

    return args, func


__all__ = ["AIFunctions", "get_openai_dict", "execute_function", "async_execute_function"]
