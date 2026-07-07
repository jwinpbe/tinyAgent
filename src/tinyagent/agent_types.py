"""Type definitions for the agent loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, TypeAlias, TypeGuard, TypeVar, Union, runtime_checkable

import msgspec
from msgspec.structs import StructMeta  # type: ignore[attr-defined]

# ------------------------------
# JSON-ish helper types
# ------------------------------

JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, Any]

ZERO_USAGE: dict[str, Any] = {
    "input": 0,
    "output": 0,
    "cache_read": 0,
    "cache_write": 0,
    "total_tokens": 0,
    "cost": {
        "input": 0.0,
        "output": 0.0,
        "cache_read": 0.0,
        "cache_write": 0.0,
        "total": 0.0,
    },
}


# ------------------------------
# Core message/content types
# ------------------------------


class _AgentBaseModel(msgspec.Struct, kw_only=True):
    """Shared base for migrated message/state models."""

    def model_dump(self, exclude_none: bool = False) -> dict[str, object]:
        result = msgspec.structs.asdict(self)
        if exclude_none:
            return {k: v for k, v in result.items() if v is not None}
        return result

    @classmethod
    def model_validate(cls, data: dict[str, object]) -> "_AgentBaseModel":
        return msgspec.convert(data, cls)


class _TaggedContentMeta(StructMeta):
    """Metaclass that enables tagged unions with `type` as the tag field."""

    def __new__(mcls, name, bases, ns, **kw):
        kw.setdefault("tag_field", "type")
        kw.setdefault("tag", True)
        return super().__new__(mcls, name, bases, ns, **kw)

    def __init__(cls, name, bases, ns, **kw):
        super().__init__(name, bases, ns, **kw)
        # Inject model_dump override to include the tag/role property.
        # Delegates to the base class method to avoid duplicating exclude_none logic.
        if "model_dump" not in ns:

            def _model_dump(self, exclude_none: bool = False) -> dict[str, object]:
                result = msgspec.to_builtins(self)
                if exclude_none:

                    def _strip_none(obj):
                        if isinstance(obj, dict):
                            return {k: _strip_none(v) for k, v in obj.items() if v is not None}
                        if isinstance(obj, list):
                            return [_strip_none(item) for item in obj]
                        return obj

                    result = _strip_none(result)
                # Include the tag field from the property (role for messages, type for content)
                for prop in ("role", "type"):
                    if hasattr(self, prop) and prop not in result:
                        result[prop] = getattr(self, prop)
                return result

            cls.model_dump = _model_dump


@runtime_checkable
class ModelDumpable(Protocol):
    """Protocol for values that support Pydantic-style model serialization."""

    def model_dump(self, *, exclude_none: bool = True) -> dict[str, object]:
        del exclude_none
        raise NotImplementedError


def dump_model_dumpable(value: object, *, where: str) -> dict[str, object]:
    """Serialize a model payload via the shared model_dump contract."""

    if not isinstance(value, ModelDumpable):
        raise TypeError(f"{where}: expected model payload with model_dump(exclude_none=True)")
    dumped = value.model_dump(exclude_none=True)
    if not isinstance(dumped, dict):
        raise TypeError(f"{where}: model_dump(exclude_none=True) must return a dict")
    return dumped


class ThinkingBudgets(_AgentBaseModel):
    """Token budgets for thinking/reasoning."""

    thinking_budget: int | None = None
    max_tokens: int | None = None


TResult = TypeVar("TResult")

MaybeAwaitable: TypeAlias = TResult | Awaitable[TResult]


class ThinkingLevel(str, Enum):
    """Thinking/reasoning level for models that support it."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class CacheControl(_AgentBaseModel):
    """Cache control directive for Anthropic prompt caching."""

    type: str | None = None


class TextContent(_AgentBaseModel, metaclass=_TaggedContentMeta, tag="text"):
    """Text content block."""

    text: str | None = None
    text_signature: str | None = None
    cache_control: CacheControl | None = None

    @property
    def type(self) -> Literal["text"]:
        return "text"


class ImageContent(_AgentBaseModel, metaclass=_TaggedContentMeta, tag="image"):
    """Image content block."""

    url: str | None = None
    mime_type: str | None = None

    @property
    def type(self) -> Literal["image"]:
        return "image"


class ThinkingContent(_AgentBaseModel, metaclass=_TaggedContentMeta, tag="thinking"):
    """Thinking content block."""

    thinking: str | None = None
    thinking_signature: str | None = None
    cache_control: CacheControl | None = None

    @property
    def type(self) -> Literal["thinking"]:
        return "thinking"


class ToolCallContent(_AgentBaseModel, metaclass=_TaggedContentMeta, tag="tool_call"):
    """Tool call content block."""

    id: str | None = None
    name: str | None = None
    arguments: dict[str, Any] = {}
    partial_json: str | None = None

    @property
    def type(self) -> Literal["tool_call"]:
        return "tool_call"


ToolCall: TypeAlias = ToolCallContent

AssistantContent: TypeAlias = TextContent | ThinkingContent | ToolCallContent


class UserMessage(_AgentBaseModel, metaclass=_TaggedContentMeta, tag="user"):
    """User message for LLM."""

    content: list[TextContent | ImageContent] = []
    timestamp: int | None = None

    @property
    def role(self) -> Literal["user"]:
        return "user"


StopReason: TypeAlias = Literal[
    "complete",
    "error",
    "aborted",
    "tool_calls",
    "stop",
    "length",
    "tool_use",
]

STOP_REASONS: frozenset[StopReason] = frozenset({
    "complete",
    "error",
    "aborted",
    "tool_calls",
    "stop",
    "length",
    "tool_use",
})


class AssistantMessage(_AgentBaseModel, metaclass=_TaggedContentMeta, tag="assistant"):
    """Assistant message from LLM."""

    content: list[AssistantContent | None] = []
    stop_reason: StopReason | None = None
    timestamp: int | None = None
    api: str | None = None
    provider: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    error_message: str | None = None

    @property
    def role(self) -> Literal["assistant"]:
        return "assistant"


class ToolResultMessage(_AgentBaseModel, metaclass=_TaggedContentMeta, tag="tool_result"):
    """Tool result message."""

    tool_call_id: str | None = None
    tool_name: str | None = None
    content: list[TextContent | ImageContent] = []
    details: dict[str, Any] = {}
    is_error: bool = False
    terminate: bool = False
    timestamp: int | None = None

    @property
    def role(self) -> Literal["tool_result"]:
        return "tool_result"

    def model_dump(self, exclude_none: bool = False) -> dict[str, object]:
        result = super().model_dump(exclude_none=exclude_none)
        result.pop("terminate", None)
        return result


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


class CustomAgentMessage(_AgentBaseModel, metaclass=_TaggedContentMeta, tag="custom"):
    """Base class for custom agent messages."""

    timestamp: int | None = None

    @property
    def role(self) -> str:
        return self.__class__.__name__

    @classmethod
    def model_validate(cls, data: dict[str, object]) -> "AgentMessage":
        # Dispatch to the right subclass based on role
        role = data.get("role", "")
        match role:
            case "user":
                return msgspec.convert(data, UserMessage)
            case "assistant":
                return msgspec.convert(data, AssistantMessage)
            case "tool_result":
                return msgspec.convert(data, ToolResultMessage)
        return super().model_validate(data)


AgentMessage = Union[Message, CustomAgentMessage]


ConvertToLlmFn: TypeAlias = Callable[[list[AgentMessage]], MaybeAwaitable[list[Message]]]
TransformContextFn: TypeAlias = Callable[
    [list[AgentMessage], asyncio.Event | None],
    Awaitable[list[AgentMessage]],
]
ApiKeyResolver: TypeAlias = Callable[[str], MaybeAwaitable[str | None]]
AgentMessageProvider: TypeAlias = Callable[[], Awaitable[list[AgentMessage]]]


# ------------------------------
# Tool types
# ------------------------------


@dataclass
class AgentToolResult:
    """Result from executing a tool."""

    content: list[TextContent | ImageContent] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    terminate: bool = False


@dataclass
class ToolLoopControl:
    """Host decision used by tool-loop control hooks."""

    terminate: bool = False
    result: AgentToolResult | None = None
    is_error: bool | None = None


AgentToolUpdateCallback = Callable[[AgentToolResult], None]


@dataclass
class Tool:
    """Tool definition."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTool(Tool):
    """Agent tool with execute function."""

    label: str = ""
    execute: Callable[..., Awaitable[AgentToolResult]] | None = None


# ------------------------------
# Context/model types
# ------------------------------


@dataclass
class Context:
    """Context for LLM calls."""

    system_prompt: str = ""
    messages: list[Message] = field(default_factory=list)
    tools: list[AgentTool] | None = None


@dataclass
class AgentContext:
    """Agent context with AgentMessage types."""

    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] | None = None


BeforeToolCallFn: TypeAlias = Callable[
    [ToolCallContent, AgentTool | None, dict[str, Any]],
    MaybeAwaitable[ToolLoopControl | None],
]
AfterToolCallFn: TypeAlias = Callable[
    [ToolCallContent, ToolResultMessage],
    MaybeAwaitable[ToolLoopControl | None],
]
ShouldStopAfterTurnFn: TypeAlias = Callable[
    [AssistantMessage, list[ToolResultMessage], AgentContext, list[AgentMessage]],
    MaybeAwaitable[bool],
]


class Model(_AgentBaseModel):
    """Model configuration."""

    provider: str = ""
    id: str = ""  # Model identifier (e.g., "gpt-4", "claude-3.5-sonnet")
    api: str = ""  # API type (e.g., "openai", "anthropic", "openrouter")
    thinking_level: ThinkingLevel = ThinkingLevel.OFF


class SimpleStreamOptions(_AgentBaseModel):
    """Standard stream options passed to providers."""

    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    signal: asyncio.Event | None = None


StreamFn: TypeAlias = Callable[[Model, Context, SimpleStreamOptions], Awaitable["StreamResponse"]]


class AssistantMessageEvent(_AgentBaseModel):
    """Event during assistant message streaming."""

    type: (
        Literal[
            "start",
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "tool_call_start",
            "tool_call_delta",
            "tool_call_end",
            "done",
            "error",
        ]
        | None
    ) = None
    partial: AssistantMessage | None = None
    content_index: int | None = None
    delta: str | None = None
    content: str | TextContent | ThinkingContent | ToolCallContent | None = None
    tool_call: ToolCallContent | None = None
    reason: str | None = None
    message: AssistantMessage | None = None
    error: AssistantMessage | str | None = None


STREAM_UPDATE_EVENTS: frozenset[str] = frozenset({
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "tool_call_start",
    "tool_call_delta",
    "tool_call_end",
})


class StreamResponse(Protocol):
    """Response from streaming."""

    def result(self) -> Awaitable[AssistantMessage]: ...

    def __aiter__(self) -> AsyncIterator[AssistantMessageEvent]: ...

    async def __anext__(self) -> AssistantMessageEvent: ...


# ------------------------------
# Agent event types
# ------------------------------


@dataclass
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass
class AgentEndEvent:
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"


@dataclass
class TurnEndEvent:
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage | None = None
    tool_results: list[ToolResultMessage] = field(default_factory=list)


@dataclass
class MessageStartEvent:
    type: Literal["message_start"] = "message_start"
    message: AgentMessage | None = None


@dataclass
class MessageUpdateEvent:
    type: Literal["message_update"] = "message_update"
    message: AgentMessage | None = None
    assistant_message_event: AssistantMessageEvent | None = None


@dataclass
class MessageEndEvent:
    type: Literal["message_end"] = "message_end"
    message: AgentMessage | None = None


@dataclass
class ToolExecutionStartEvent:
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] | None = None


@dataclass
class ToolExecutionUpdateEvent:
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] | None = None
    partial_result: AgentToolResult | None = None


@dataclass
class ToolExecutionEndEvent:
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str = ""
    tool_name: str = ""
    result: AgentToolResult | None = None
    is_error: bool = False


AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
]


MessageEvent = MessageStartEvent | MessageUpdateEvent | MessageEndEvent
ToolExecutionEvent = ToolExecutionStartEvent | ToolExecutionUpdateEvent | ToolExecutionEndEvent


def is_agent_end_event(event: AgentEvent) -> TypeGuard[AgentEndEvent]:
    return isinstance(event, AgentEndEvent)


def is_turn_end_event(event: AgentEvent) -> TypeGuard[TurnEndEvent]:
    return isinstance(event, TurnEndEvent)


def is_message_start_or_update_event(
    event: AgentEvent,
) -> TypeGuard[MessageStartEvent | MessageUpdateEvent]:
    return isinstance(event, MessageStartEvent | MessageUpdateEvent)


def is_message_end_event(event: AgentEvent) -> TypeGuard[MessageEndEvent]:
    return isinstance(event, MessageEndEvent)


def is_message_event(event: AgentEvent) -> TypeGuard[MessageEvent]:
    return isinstance(event, MessageStartEvent | MessageUpdateEvent | MessageEndEvent)


def is_tool_execution_start_event(event: AgentEvent) -> TypeGuard[ToolExecutionStartEvent]:
    return isinstance(event, ToolExecutionStartEvent)


def is_tool_execution_end_event(event: AgentEvent) -> TypeGuard[ToolExecutionEndEvent]:
    return isinstance(event, ToolExecutionEndEvent)


def is_tool_execution_event(event: AgentEvent) -> TypeGuard[ToolExecutionEvent]:
    return isinstance(
        event,
        ToolExecutionStartEvent | ToolExecutionUpdateEvent | ToolExecutionEndEvent,
    )


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop."""

    model: Model
    convert_to_llm: ConvertToLlmFn
    transform_context: TransformContextFn | None = None
    get_api_key: ApiKeyResolver | None = None
    get_steering_messages: AgentMessageProvider | None = None
    get_follow_up_messages: AgentMessageProvider | None = None
    before_tool_call: BeforeToolCallFn | None = None
    after_tool_call: AfterToolCallFn | None = None
    should_stop_after_turn: ShouldStopAfterTurnFn | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class AgentState(_AgentBaseModel):
    """Agent state containing all configuration and conversation data."""

    system_prompt: str = ""
    model: Model | None = None
    thinking_level: ThinkingLevel = ThinkingLevel.OFF
    tools: list[AgentTool] = []
    messages: list[AgentMessage] = []
    is_streaming: bool = False
    stream_message: AgentMessage | None = None
    pending_tool_calls: set[str] = set()
    error: str | None = None


@dataclass(frozen=True)
class _WakeupSignal:
    """Internal queue marker used to wake blocked stream consumers."""


EventStreamQueueItem: TypeAlias = AgentEvent | _WakeupSignal


class EventStream:
    """Async event stream that yields events and returns a final result.

    Important: Some agent loops run in background tasks via `asyncio.create_task()`.
    If that background task fails, we must propagate the exception to the consumer
    of the stream. Otherwise callers awaiting `agent.prompt()` can hang forever.
    """

    _WAKEUP_SENTINEL = _WakeupSignal()

    def __init__(
        self,
        is_end_event: Callable[[AgentEvent], bool],
        get_result: Callable[[AgentEvent], list[AgentMessage]],
    ):
        self._queue: asyncio.Queue[EventStreamQueueItem] = asyncio.Queue()
        self._is_end_event = is_end_event
        self._get_result = get_result
        self._result: list[AgentMessage] | None = None
        self._ended = False
        self._exception: BaseException | None = None

    def push(self, event: AgentEvent) -> None:
        if self._ended:
            return
        self._queue.put_nowait(event)

    def end(self, result: list[AgentMessage]) -> None:
        self._result = result
        self._ended = True
        self._queue.put_nowait(self._WAKEUP_SENTINEL)

    def set_exception(self, exc: BaseException) -> None:
        """Terminate the stream with an exception.

        The next consumer read will raise `exc` once all already-queued events
        are drained.
        """

        if self._ended:
            return
        self._exception = exc
        self._ended = True
        self._queue.put_nowait(self._WAKEUP_SENTINEL)

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        return self

    async def __anext__(self) -> AgentEvent:
        while True:
            if self._queue.empty():
                if self._exception is not None:
                    exc = self._exception
                    self._exception = None
                    raise exc
                if self._ended:
                    raise StopAsyncIteration

            queued_item = await self._queue.get()
            if isinstance(queued_item, _WakeupSignal):
                if self._exception is not None:
                    exc = self._exception
                    self._exception = None
                    raise exc
                if self._ended:
                    raise StopAsyncIteration
                continue

            event = queued_item
            if self._is_end_event(event):
                self._result = self._get_result(event)
                self._ended = True
            return event

    async def result(self) -> list[AgentMessage]:
        while True:
            if self._queue.empty() and self._exception is not None:
                exc = self._exception
                self._exception = None
                raise exc
            if self._ended and self._queue.empty():
                break
            try:
                await self.__anext__()
            except StopAsyncIteration:
                break

        return self._result or []
