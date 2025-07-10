
import sys
import re
import json
import argparse
import asyncio
import signal
import socket
import yaml
import functools
import os
from typing import Any, Dict, List, Optional, Sequence, Union, Callable
from collections import defaultdict
from http import HTTPStatus
from ssl import SSLContext
import uvicorn
from fastapi import FastAPI, Request, Response
from vllm import envs
from vllm.engine.async_llm_engine import AsyncEngineDeadError
from vllm.engine.multiprocessing import MQEngineDeadError
from vllm.engine.protocol import EngineClient
from vllm.utils import find_process_using_port
#from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from watchfiles import Change, awatch 
from vllm.platforms import current_platform
from vllm.logger import init_logger
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs as vllmAsync
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs 
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask, BackgroundTasks
from vllm.utils import print_warning_once
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest)

logger = init_logger(__name__)

__all__ = [
    'serve_http', 'FlexibleArgumentParser', 'set_ulimit', 'with_cancellation', 
    'init_logger', 'AsyncLLMEngine', '_AsyncLLMEngine', 'AsyncEngineArgs'
]  
class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""
    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context

class SSLCertRefresher:
    """A class that monitors SSL certificate files and reloads them when they change."""

    def __init__(self,
                 ssl_context: SSLContext,
                 key_path: Optional[str] = None,
                 cert_path: Optional[str] = None,
                 ca_path: Optional[str] = None) -> None:
        self.ssl = ssl_context
        self.key_path = key_path
        self.cert_path = cert_path
        self.ca_path = ca_path

        # Setup certification chain watcher
        def update_ssl_cert_chain(change: Change, file_path: str) -> None:
            logger.info("Reloading SSL certificate chain")
            assert self.key_path and self.cert_path
            self.ssl.load_cert_chain(self.cert_path, self.key_path)

        self.watch_ssl_cert_task = None
        if self.key_path and self.cert_path:
            self.watch_ssl_cert_task = asyncio.create_task(
                self._watch_files([self.key_path, self.cert_path],
                                  update_ssl_cert_chain))

        # Setup CA files watcher
        def update_ssl_ca(change: Change, file_path: str) -> None:
            logger.info("Reloading SSL CA certificates")
            assert self.ca_path
            self.ssl.load_verify_locations(self.ca_path)

        self.watch_ssl_ca_task = None
        if self.ca_path:
            self.watch_ssl_ca_task = asyncio.create_task(
                self._watch_files([self.ca_path], update_ssl_ca))

    async def _watch_files(self, paths, fun: Callable[[Change, str], None]) -> None:
        """Watch multiple file paths asynchronously."""
        logger.info("SSLCertRefresher monitors files: %s", paths)
        async for changes in awatch(*paths):
            try:
                for change, file_path in changes:
                    logger.info("File change detected: %s - %s", change.name,
                                file_path)
                    fun(change, file_path)
            except Exception as e:
                logger.error(
                    "SSLCertRefresher failed taking action on file change. "
                    "Error: %s", e)

    def stop(self) -> None:
        """Stop watching files."""
        if self.watch_ssl_cert_task:
            self.watch_ssl_cert_task.cancel()
            self.watch_ssl_cert_task = None
        if self.watch_ssl_ca_task:
            self.watch_ssl_ca_task.cancel()
            self.watch_ssl_ca_task = None

# Serve HTTP API with uvicorn
async def serve_http(app: FastAPI,
                     sock: Optional[socket.socket],
                     enable_ssl_refresh: bool = False,
                     **uvicorn_kwargs: Any):
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ', '.join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    config.load()
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    watchdog_task = loop.create_task(
        watchdog_loop(server, app.state.engine_client))
    server_task = loop.create_task(
        server.serve(sockets=[sock] if sock else None))

    # Handle SSL certificate refreshing based on configuration
    ssl_cert_refresher = None
    if enable_ssl_refresh:
        ssl_cert_refresher = SSLCertRefresher(
            ssl_context=config.ssl,
            key_path=config.ssl_keyfile,
            cert_path=config.ssl_certfile,
            ca_path=config.ssl_ca_certs
        )

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()
        watchdog_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.debug(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()
    finally:
        watchdog_task.cancel()

async def watchdog_loop(server: uvicorn.Server, engine: EngineClient):
    """
    Watchdog task that runs in the background, checking
    for error state in the engine. Needed to trigger shutdown
    if an exception arises in the StreamingResponse() generator.
    """
    VLLM_WATCHDOG_TIME_S = 5.0
    while True:
        await asyncio.sleep(VLLM_WATCHDOG_TIME_S)
        terminate_if_errored(server, engine)

def terminate_if_errored(server: uvicorn.Server, engine: EngineClient):
    """
    Checks if the engine has errored and triggers the shutdown if needed.
    """
    engine_errored = engine.errored and not engine.is_running
    if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH and engine_errored:
        server.should_exit = True

def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server) -> None:
    """
    Adds exception handlers for errors in the LLM engine like EngineDeadError
    and EngineGenerateError.
    """
    @app.exception_handler(RuntimeError)
    @app.exception_handler(AsyncEngineDeadError)
    @app.exception_handler(MQEngineDeadError)
    @app.exception_handler(EngineDeadError)
    @app.exception_handler(EngineGenerateError)
    async def runtime_exception_handler(request: Request, __):
        terminate_if_errored(
            server=server,
            engine=request.app.state.engine_client,
        )

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)




class StoreBoolean(argparse.Action):
    """Action to store boolean values from command-line flags."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)

class SortedHelpFormatter(argparse.HelpFormatter):
    """Sort arguments alphabetically in help output."""
    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda a: a.option_strings)
        super().add_arguments(actions)

class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""
    
    _deprecated_actions: set[argparse.Action] = set()

    def __init__(self, *args, **kwargs):
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = SortedHelpFormatter
        super().__init__(*args, **kwargs)

    if sys.version_info < (3, 13):
        def parse_known_args(self, args=None, namespace=None):
            namespace, args = super().parse_known_args(args, namespace)
            for action in self._deprecated_actions:
                dest = action.dest
                if hasattr(namespace, dest) and getattr(namespace, dest) != action.default:
                    print_warning_once(f"Argument '{dest}' is deprecated.")
            return namespace, args

        def add_argument(self, *args, **kwargs):
            deprecated = kwargs.pop("deprecated", False)
            action = super().add_argument(*args, **kwargs)
            if deprecated:
                self._deprecated_actions.add(action)
            return action

        class _FlexibleArgumentGroup(argparse._ArgumentGroup):
            def add_argument(self, *args, **kwargs):
                deprecated = kwargs.pop("deprecated", False)
                action = super().add_argument(*args, **kwargs)
                if deprecated:
                    FlexibleArgumentParser._deprecated_actions.add(action)
                return action

        def add_argument_group(self, *args, **kwargs):
            group = self._FlexibleArgumentGroup(self, *args, **kwargs)
            self._action_groups.append(group)
            return group

    def parse_args(  # type: ignore[override]
        self,
        args: Optional[List[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ):
        if args is None:
            args = sys.argv[1:]

        if args and args[0] == "serve":
            model_in_cli = len(args) > 1 and not args[1].startswith('-')
            model_in_config = any(arg == '--model' for arg in args)
            if not model_in_cli and not model_in_config:
                raise ValueError(
                    "No model specified! Please specify model either "
                    "as a positional argument or in a config file.")
            if '--model' in args:
                raise ValueError(
                    "With `vllm serve`, you should provide the model as a "
                    "positional argument or in a config file instead of via "
                    "the `--model` option.")

        if '--config' in args:
            args = self._pull_args_from_config(args)

        def repl(match: re.Match) -> str:
            return match.group(0).replace("_", "-")

        pattern = re.compile(r"(?<=--)[^\.]*")
        processed_args = []

        for i, arg in enumerate(args):
            if arg.startswith('--'):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = pattern.sub(repl, key, count=1)
                    processed_args.append(f'{key}={value}')
                else:
                    key = pattern.sub(repl, arg, count=1)
                    processed_args.append(key)
            elif arg.startswith('-O') and arg != '-O' and arg[2] != '.':
                level = arg[3:] if arg[2] == '=' else arg[2:]
                processed_args.append(f'-O.level={level}')
            elif arg == '-O' and i + 1 < len(args) and args[i + 1] in {"0", "1", "2", "3"}:
                processed_args.append('-O.level')
            else:
                processed_args.append(arg)

        def create_nested_dict(keys: List[str], value: str) -> Dict[str, Any]:
            nested_dict: Any = value
            for key in reversed(keys):
                nested_dict = {key: nested_dict}
            return nested_dict

        def recursive_dict_update(
            original: Dict[str, Any],
            update: Dict[str, Any],
        ) -> set[str]:
            duplicates = set()
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(original.get(k), dict):
                    nested_duplicates = recursive_dict_update(original[k], v)
                    duplicates.update(f"{k}.{d}" for d in nested_duplicates)
                elif isinstance(v, list) and isinstance(original.get(k), list):
                    original[k] += v
                else:
                    if k in original:
                        duplicates.add(k)
                    original[k] = v
            return duplicates

        delete = set()
        dict_args = defaultdict(dict)
        duplicates = set()

        for i, arg in enumerate(processed_args):
            if i in delete:
                continue
            if arg.startswith("-") and "." in arg:
                if "=" in arg:
                    key_part, value_str = arg.split("=", 1)
                else:
                    key_part = arg
                    value_str = processed_args[i + 1]
                    delete.add(i + 1)
                if key_part.endswith("+"):
                    key_part = key_part[:-1]
                    value_str = json.dumps(list(value_str.split(",")))
                key, *keys = key_part.split(".")
                try:
                    value = json.loads(value_str)
                except json.JSONDecodeError:
                    value = value_str
                arg_dict = create_nested_dict(keys, value)
                arg_duplicates = recursive_dict_update(dict_args[key], arg_dict)
                duplicates.update(f'{key}.{d}' for d in arg_duplicates)
                delete.add(i)

        processed_args = [a for i, a in enumerate(processed_args) if i not in delete]

        if duplicates:
            logger.warning("Found duplicate keys: %s", ", ".join(duplicates))

        for key, value in dict_args.items():
            processed_args.append(key)
            processed_args.append(json.dumps(value))

        return super().parse_args(processed_args, namespace)

    def check_port(self, value):
        try:
            value = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Port must be an integer")
        if not (1024 <= value <= 65535):
            raise argparse.ArgumentTypeError("Port must be between 1024 and 65535")
        return value

    def _pull_args_from_config(self, args: List[str]) -> List[str]:
        assert args.count('--config') <= 1, "More than one config file specified!"
        index = args.index('--config')
        if index == len(args) - 1:
            raise ValueError("No config file specified!")
        file_path = args[index + 1]
        config_args = self._load_config_file(file_path)
        if args[0] == "serve":
            model_in_cli = len(args) > 1 and not args[1].startswith('-')
            model_in_config = any(arg == '--model' for arg in config_args)
            if not model_in_cli and not model_in_config:
                raise ValueError("No model specified in CLI or config.")
            if model_in_cli:
                args = [args[0], args[1]] + config_args + args[2:index] + args[index + 2:]
            else:
                args = [args[0]] + config_args + args[1:index] + args[index + 2:]
        else:
            args = [args[0]] + config_args + args[1:index] + args[index + 2:]
        return args

    def _load_config_file(self, file_path: str) -> List[str]:
        extension = file_path.split('.')[-1]
        if extension not in ('yaml', 'yml'):
            raise ValueError(f"Config file must be of type yaml/yml. Got {extension}")
        config = {}
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as ex:
            logger.error(f"Unable to read config file at {file_path}")
            raise ex
        processed_args = []
        store_boolean_arguments = [action.dest for action in self._actions if isinstance(action, StoreBoolean)]
        for key, value in config.items():
            if isinstance(value, bool) and key not in store_boolean_arguments:
                if value:
                    processed_args.append(f'--{key}')
            else:
                processed_args.append(f'--{key}')
                processed_args.append(str(value))
        return processed_args

def set_ulimit(target_soft_limit=65535):
    if sys.platform.startswith('win'):
        logger.info("Windows detected, skipping ulimit adjustment.")
        return

    import resource
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type,
                               (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n", current_soft, e)

VLLM_SUBCMD_PARSER_EPILOG = (
    "Tip: Use `vllm [serve|run-batch|bench <bench_type>] "
    "--help=<keyword>` to explore arguments from help.\n"
    "   - To view a argument group:     --help=ModelConfig\n"
    "   - To view a single argument:    --help=max-num-seqs\n"
    "   - To search by keyword:         --help=max\n"
    "   - To list all groups:           --help=listgroup")


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            if request.app.state.enable_server_load_tracking:
                # on timeout/cancellation the BackgroundTask in load_aware_call
                # cannot decrement the server load metrics.
                # Must be decremented by with_cancellation instead.
                request.app.state.server_load_metrics -= 1
            break


def with_cancellation(handler_func):
    """Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for a http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.
    """

    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait([handler_task, cancellation_task],
                                           return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def decrement_server_load(request: Request):
    request.app.state.server_load_metrics -= 1


def load_aware_call(func):

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        raw_request = kwargs.get("raw_request", args[1] if len(args) > 1 else None)

        if raw_request is None:
            raise ValueError(
                "raw_request required when server load tracking is enabled")

        if not raw_request.app.state.enable_server_load_tracking:
            return await func(*args, **kwargs)

        raw_request.app.state.server_load_metrics += 1
        try:
            response = await func(*args, **kwargs)
        except Exception:
            raw_request.app.state.server_load_metrics -= 1
            raise

        if isinstance(response, (JSONResponse, StreamingResponse)):
            if response.background is None:
                response.background = BackgroundTask(decrement_server_load,
                                                     raw_request)
            elif isinstance(response.background, BackgroundTasks):
                response.background.add_task(decrement_server_load,
                                             raw_request)
            elif isinstance(response.background, BackgroundTask):
                # Convert the single BackgroundTask to BackgroundTasks
                # and chain the decrement_server_load task to it
                tasks = BackgroundTasks()
                tasks.add_task(response.background.func,
                               *response.background.args,
                               **response.background.kwargs)
                tasks.add_task(decrement_server_load, raw_request)
                response.background = tasks
        else:
            raw_request.app.state.server_load_metrics -= 1

        return response

    return wrapper


def cli_env_setup():
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logger.debug("Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _validate_truncation_size(
    max_model_len: int,
    truncate_prompt_tokens: Optional[int],
    tokenization_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[int]:

    if truncate_prompt_tokens is not None:
        if truncate_prompt_tokens <= -1:
            truncate_prompt_tokens = max_model_len

        if truncate_prompt_tokens > max_model_len:
            raise ValueError(
                f"truncate_prompt_tokens value ({truncate_prompt_tokens}) "
                f"is greater than max_model_len ({max_model_len})."
                f" Please, select a smaller truncation size.")

        if tokenization_kwargs is not None:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = truncate_prompt_tokens

    else:
        if tokenization_kwargs is not None:
            tokenization_kwargs["truncation"] = False

    return truncate_prompt_tokens


def show_filtered_argument_or_group_from_help(parser: argparse.ArgumentParser,
                                              subcommand_name: list[str]):

    if len(sys.argv) <= len(subcommand_name) or sys.argv[
            1:1 + len(subcommand_name)] != subcommand_name:
        return

    for arg in sys.argv:
        if arg.startswith('--help='):
            search_keyword = arg.split('=', 1)[1]

            if search_keyword == 'listgroup':
                print("\nAvailable argument groups:")
                for group in parser._action_groups:
                    if group.title and not group.title.startswith(
                            "positional arguments"):
                        print(f"  - {group.title}")
                        if group.description:
                            print("    " + group.description.strip())
                        print()
                sys.exit(0)

            formatter = parser._get_formatter()
            for group in parser._action_groups:
                if group.title and group.title.lower() == search_keyword.lower():
                    formatter.start_section(group.title)
                    formatter.add_text(group.description)
                    formatter.add_arguments(group._group_actions)
                    formatter.end_section()
                    print(formatter.format_help())
                    sys.exit(0)

            matched_actions = []

            for group in parser._action_groups:
                for action in group._group_actions:
                    if any(search_keyword.lower() in opt.lower()
                           for opt in action.option_strings):
                        matched_actions.append(action)

            if matched_actions:
                print(f"\nParameters matching '{search_keyword}':\n")
                formatter = parser._get_formatter()
                formatter.add_arguments(matched_actions)
                print(formatter.format_help())
                sys.exit(0)

            print(f"\nNo group or parameter matching '{search_keyword}'")
            print("Tip: use `--help=listgroup` to view all groups.")
            sys.exit(1)


def get_max_tokens(max_model_len: int, request: Union[ChatCompletionRequest,
                                                      CompletionRequest],
                   input_length: int, default_sampling_params: dict) -> int:

    # Ensure the request object is valid and contains necessary parameters
    max_tokens = getattr(request, "max_completion_tokens", None) or request.max_tokens
    default_max_tokens = max_model_len - input_length
    max_output_tokens = current_platform.get_max_output_tokens(input_length)

    return min(val
               for val in (default_max_tokens, max_tokens, max_output_tokens,
                           default_sampling_params.get("max_tokens"))
               if val is not None)