"""Type definitions for the agent loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import Enum
from typing import Any, Literal, Protocol, TypeAlias, TypeGuard, TypeVar, Union, runtime_checkable

import msgspec

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


def _strip_none_recursive(obj: object) -> object:
    """Recursively remove None values from dicts and lists."""
    if isinstance(obj, dict):
        return {k: _strip_none_recursive(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none_recursive(item) for item in obj]
    return obj


def _tagged_model_dump(
    struct: msgspec.Struct, exclude_none: bool
) -> dict[str, object]:
    """model_dump for tagged structs: uses to_builtins so the tag field
    is included in the serialized output (required for wire-format stability)."""
    result: dict[str, object] = msgspec.to_builtins(struct)
    if exclude_none:
        result = _strip_none_recursive(result)  # type: ignore[assignment]
    return result


class _AgentBaseModel(msgspec.Struct, kw_only=True):
    """Shared base for migrated message/state models.

    Non-tagged structs use asdict (preserves nested structs, tolerates
    non-serializable fields like asyncio.Event).  Tagged subclasses
    override model_dump to use to_builtins so the discriminant tag is
    emitted.
    """

    def model_dump(self, exclude_none: bool = False) -> dict[str, object]:
        result = msgspec.structs.asdict(self)
        if exclude_none:
            return {k: v for k, v in result.items() if v is not None}
        return result

    @classmethod
    def model_validate(cls, data: dict[str, object]) -> _AgentBaseModel:
        return msgspec.convert(data, cls)


class _TaggedContent(_AgentBaseModel, tag_field="type", tag=True):
    """Base for content blocks tagged by ``type`` field."""

    @property
    def type(self) -> str:
        """Return the content block's type tag (e.g. "text", "image")."""
        return self.__class__.__struct_config__.tag  # type: ignore[return-value]

    def model_dump(self, exclude_none: bool = False) -> dict[str, object]:
        return _tagged_model_dump(self, exclude_none)


class _TaggedMessage(_AgentBaseModel, tag_field="role", tag=True):
    """Base for message types tagged by ``role`` field."""

    @property
    def role(self) -> str:
        """Return the message's role tag (e.g. "user", "assistant")."""
        return self.__class__.__struct_config__.tag  # type: ignore[return-value]

    def model_dump(self, exclude_none: bool = False) -> dict[str, object]:
        return _tagged_model_dump(self, exclude_none)


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


class TextContent(_TaggedContent, tag="text"):
    """Text content block."""

    text: str | None = None
    text_signature: str | None = None
    cache_control: CacheControl | None = None


class ImageContent(_TaggedContent, tag="image"):
    """Image content block."""

    url: str | None = None
    mime_type: str | None = None


class ThinkingContent(_TaggedContent, tag="thinking"):
    """Thinking content block."""

    thinking: str | None = None
    thinking_signature: str | None = None
    cache_control: CacheControl | None = None


class ToolCallContent(_TaggedContent, tag="tool_call"):
    """Tool call content block."""

    id: str | None = None
    name: str | None = None
    arguments: dict[str, Any] = {}
    partial_json: str | None = None


ToolCall: TypeAlias = ToolCallContent

AssistantContent: TypeAlias = TextContent | ThinkingContent | ToolCallContent | ImageContent


class UserMessage(_TaggedMessage, tag="user"):
    """User message for LLM."""

    content: list[TextContent | ImageContent] = []
    timestamp: int | None = None


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


class AssistantMessage(_TaggedMessage, tag="assistant"):
    """Assistant message from LLM."""

    content: list[AssistantContent | None] = []
    stop_reason: StopReason | None = None
    timestamp: int | None = None
    api: str | None = None
    provider: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    error_message: str | None = None


class ToolResultMessage(_TaggedMessage, tag="tool_result"):
    """Tool result message."""

    tool_call_id: str | None = None
    tool_name: str | None = None
    content: list[TextContent | ImageContent] = []
    details: dict[str, Any] = {}
    is_error: bool = False
    terminate: bool = False
    timestamp: int | None = None

    def model_dump(self, exclude_none: bool = False) -> dict[str, object]:
        result = super().model_dump(exclude_none=exclude_none)
        result.pop("terminate", None)
        return result


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


class CustomAgentMessage(_TaggedMessage, tag="custom"):
    """Base class for custom agent messages."""

    timestamp: int | None = None

    @property
    def role(self) -> str:
        """Return the class name as role, matching pre-refactor behavior."""
        return self.__class__.__name__

    @classmethod
    def model_validate(cls, data: dict[str, object]) -> "AgentMessage":
        # Dispatch to the right subclass based on role tag
        role = data.get("role", "")
        match role:
            case "user":
                return msgspec.convert(data, UserMessage)
            case "assistant":
                return msgspec.convert(data, AssistantMessage)
            case "tool_result":
                return msgspec.convert(data, ToolResultMessage)
        return super().model_validate(data)  # type: ignore[return-value]


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


class AgentToolResult(_AgentBaseModel):
    """Result from executing a tool."""

    content: list[TextContent | ImageContent] = []
    details: dict[str, Any] = {}
    terminate: bool = False


class ToolLoopControl(_AgentBaseModel):
    """Host decision used by tool-loop control hooks."""

    terminate: bool = False
    result: AgentToolResult | None = None
    is_error: bool | None = None


AgentToolUpdateCallback = Callable[[AgentToolResult], None]


class Tool(_AgentBaseModel):
    """Tool definition."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}


class AgentTool(Tool):
    """Agent tool with execute function."""

    label: str = ""
    execute: Callable[..., Awaitable[AgentToolResult]] | None = None


# ------------------------------
# Context/model types
# ------------------------------


class Context(_AgentBaseModel):
    """Context for LLM calls."""

    system_prompt: str = ""
    messages: list[Message] = []
    tools: list[AgentTool] | None = None


class AgentContext(_AgentBaseModel):
    """Agent context with AgentMessage types."""

    system_prompt: str = ""
    messages: list[AgentMessage] = []
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


class AgentStartEvent(_AgentBaseModel):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(_AgentBaseModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = []


class TurnStartEvent(_AgentBaseModel):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(_AgentBaseModel):
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage | None = None
    tool_results: list[ToolResultMessage] = []


class MessageStartEvent(_AgentBaseModel):
    type: Literal["message_start"] = "message_start"
    message: AgentMessage | None = None


class MessageUpdateEvent(_AgentBaseModel):
    type: Literal["message_update"] = "message_update"
    message: AgentMessage | None = None
    assistant_message_event: AssistantMessageEvent | None = None


class MessageEndEvent(_AgentBaseModel):
    type: Literal["message_end"] = "message_end"
    message: AgentMessage | None = None


class ToolExecutionStartEvent(_AgentBaseModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] | None = None


class ToolExecutionUpdateEvent(_AgentBaseModel):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] | None = None
    partial_result: AgentToolResult | None = None


class ToolExecutionEndEvent(_AgentBaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str = ""
    tool_name: str = ""
    result: AgentToolResult | None = None
    is_error: bool = False
    args: dict[str, Any] | None = None


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


class AgentLoopConfig(_AgentBaseModel):
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


class _WakeupSignal(msgspec.Struct, frozen=True):
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
