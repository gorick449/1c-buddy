from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

import httpx

from ..config import Settings, get_settings
from ..onec_models import (
    ApiError,
    ConversationRequest,
    ConversationResponse,
    MessageChunk,
    MessageRequest,
    ToolResultItem,
    ToolResultRequest,
)

logger = logging.getLogger(__name__)
_RE_REASONING_BLOCK = re.compile(r"<reasoning>.*?</reasoning>\s*", re.DOTALL)

# Транзиентный ConnectTimeout до апстрима (code.1c.ai) — основная причина
# периодических -32603. Эти ошибки возникают ДО отправки байтов запроса, поэтому
# повтор безопасен (идемпотентен). Ошибки чтения/стрима тут НЕ ретраятся —
# сообщение могло быть частично отправлено.
_CONNECT_RETRY_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)
_CONNECT_MAX_ATTEMPTS = 3
_CONNECT_RETRY_BACKOFF = 0.5  # секунды, умножается на номер попытки


class McpUpstreamToolsClient:
    """MCP-only upstream client. Must not be used by chat/OpenAI paths."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings: Settings = settings or get_settings()
        self.base_url = self.settings.ONEC_AI_BASE_URL.rstrip("/")
        self._last_assistant_uuid: Dict[str, Optional[str]] = {}
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.settings.ONEC_AI_TIMEOUT,
                read=None,
                write=self.settings.ONEC_AI_TIMEOUT,
                pool=self.settings.ONEC_AI_TIMEOUT,
            ),
            headers={
                "Accept": "*/*",
                "Accept-Charset": "utf-8",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "ru-ru,en-us;q=0.8,en;q=0.7",
                "Authorization": self.settings.ONEC_AI_TOKEN,
                "Content-Type": "application/json; charset=utf-8",
                "Origin": self.settings.ONEC_AI_BASE_URL,
                "Referer": f"{self.settings.ONEC_AI_BASE_URL}/chat/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/620.1 (KHTML, like Gecko) JavaFX/22 Safari/620.1",
            },
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def _retry_on_connect(self, factory, *, what: str):
        """Повторить корутину-фабрику при транзиентных ошибках подключения к апстриму.

        ConnectTimeout/ConnectError/PoolTimeout происходят до отправки байтов
        запроса, поэтому повторный запуск factory безопасен. Прочие ошибки
        (ReadError, RemoteProtocolError и т.п.) пробрасываются без повтора.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, _CONNECT_MAX_ATTEMPTS + 1):
            try:
                return await factory()
            except _CONNECT_RETRY_EXCEPTIONS as e:
                last_exc = e
                logger.warning(
                    "MCP upstream connect error on %s (attempt %d/%d): %r",
                    what, attempt, _CONNECT_MAX_ATTEMPTS, e,
                )
                if attempt < _CONNECT_MAX_ATTEMPTS:
                    await asyncio.sleep(_CONNECT_RETRY_BACKOFF * attempt)
        raise ApiError(
            f"Upstream connection failed after {_CONNECT_MAX_ATTEMPTS} attempts "
            f"on {what}: {last_exc!r}"
        )

    async def create_conversation(
        self, programming_language: Optional[str] = None
    ) -> str:
        request_data = ConversationRequest(
            is_chat=True,
            skill_name="custom",
            ui_language=self.settings.ONEC_AI_UI_LANGUAGE,
            programming_language=programming_language
            or self.settings.ONEC_AI_PROGRAMMING_LANGUAGE,
        )
        url = f"{self.base_url}/chat_api/v1/conversations/"
        logger.debug(
            "MCP upstream create conversation: url=%s payload=%s",
            url,
            request_data.model_dump(),
        )
        async def _do_post():
            return await self.client.post(
                url,
                json=request_data.model_dump(),
                headers={"Session-Id": ""},
            )

        try:
            resp = await self._retry_on_connect(
                _do_post, what="create_conversation"
            )
        except httpx.RequestError as e:
            raise ApiError(f"Network error creating conversation: {str(e)}")

        if resp.status_code != 200:
            raise ApiError(
                f"Conversation create error: {resp.status_code}",
                resp.status_code,
                data=self._extract_upstream_error_data(resp),
            )

        conv = ConversationResponse(**resp.json())
        self._last_assistant_uuid[conv.uuid] = None
        return conv.uuid

    def _remember_assistant_uuid(
        self, conversation_id: str, assistant_uuid: Optional[str]
    ) -> None:
        if assistant_uuid:
            self._last_assistant_uuid[conversation_id] = assistant_uuid

    @staticmethod
    def _extract_upstream_error_data(response: httpx.Response) -> Dict[str, Any]:
        data: Dict[str, Any] = {"upstream_status": response.status_code}
        request_id = response.headers.get("X-Request-ID")
        if request_id:
            data["upstream_request_id"] = request_id
        text = (response.text or "").strip()
        if not text:
            return data

        try:
            parsed = response.json()
        except Exception:
            data["detail"] = text[:2000]
            return data

        if isinstance(parsed, dict):
            error_type = parsed.get("error_type")
            if isinstance(error_type, str) and error_type.strip():
                data["upstream_error_type"] = error_type.strip()

            raw_error = parsed.get("error")
            if isinstance(raw_error, list):
                messages = []
                for item in raw_error:
                    if not isinstance(item, dict):
                        continue
                    msg = item.get("msg")
                    if isinstance(msg, str) and msg.strip():
                        messages.append(msg.strip())
                if messages:
                    data["detail"] = " | ".join(messages)
                data["upstream_error"] = raw_error
            elif isinstance(raw_error, dict):
                data["upstream_error"] = raw_error
                msg = raw_error.get("msg")
                if isinstance(msg, str) and msg.strip():
                    data["detail"] = msg.strip()
            elif raw_error is not None:
                data["upstream_error"] = raw_error

            if "detail" not in data:
                message = parsed.get("message")
                if isinstance(message, str) and message.strip():
                    data["detail"] = message.strip()
                else:
                    data["detail"] = text[:2000]
            return data

        data["detail"] = text[:2000]
        return data

    async def _send_user_message(
        self,
        conversation_id: str,
        instruction: str,
        parent_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = MessageRequest.from_instruction(
            instruction, parent_uuid=parent_uuid
        ).model_dump()
        return await self._collect_stream(conversation_id, payload)

    async def call_prompt(
        self,
        conversation_id: str,
        *,
        instruction: str,
        parent_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = await self._send_user_message(
            conversation_id,
            instruction,
            parent_uuid=parent_uuid,
        )
        aggregated_result: Dict[str, Any] = {
            "assistant_uuid": result.get("assistant_uuid"),
            "final_text": result.get("final_text") or "",
            "full_text": result.get("full_text") or "",
            "tool_calls": list(result.get("tool_calls") or []),
            "tool_results": list(result.get("tool_results") or []),
            "tool_followups": list(result.get("tool_followups") or []),
        }

        while True:
            tool_calls = result.get("tool_calls") or []
            assistant_uuid = result.get("assistant_uuid")
            if not tool_calls or not assistant_uuid:
                aggregated_result["assistant_uuid"] = (
                    result.get("assistant_uuid")
                    or aggregated_result.get("assistant_uuid")
                )
                aggregated_result["final_text"] = (
                    result.get("final_text")
                    or aggregated_result.get("final_text")
                    or ""
                )
                aggregated_result["full_text"] = self._merge_full_text(
                    aggregated_result.get("full_text") or "",
                    result.get("full_text") or result.get("final_text") or "",
                )
                aggregated_result["tool_calls"] = list(tool_calls)
                return aggregated_result

            next_call = tool_calls[-1]
            result = await self._respond_to_tool_call(
                conversation_id,
                assistant_uuid=assistant_uuid,
                tool_call=next_call,
                ensure_ascii=False,
            )
            aggregated_result["assistant_uuid"] = (
                result.get("assistant_uuid")
                or aggregated_result.get("assistant_uuid")
            )
            aggregated_result["final_text"] = (
                result.get("final_text")
                or aggregated_result.get("final_text")
                or ""
            )
            aggregated_result["full_text"] = self._merge_full_text(
                aggregated_result.get("full_text") or "",
                result.get("full_text") or result.get("final_text") or "",
            )
            aggregated_result["tool_calls"] = list(result.get("tool_calls") or [])
            aggregated_result["tool_results"].extend(result.get("tool_results") or [])
            aggregated_result["tool_followups"].extend(
                result.get("tool_followups") or []
            )

    async def _ensure_assistant_parent_uuid(
        self, conversation_id: str, parent_uuid: Optional[str] = None
    ) -> str:
        if parent_uuid:
            return parent_uuid
        last_uuid = self._last_assistant_uuid.get(conversation_id)
        if last_uuid:
            return last_uuid

        bootstrap = await self._send_user_message(
            conversation_id, "Ответь одним словом: Готово."
        )
        assistant_uuid = bootstrap.get("assistant_uuid")
        if not assistant_uuid:
            raise ApiError("Unable to obtain assistant parent UUID for MCP tool call")
        return assistant_uuid

    def _build_tool_ack_payload(
        self,
        *,
        parent_uuid: str,
        tool_call_id: str,
    ) -> Dict[str, Any]:
        request = ToolResultRequest(
            parent_uuid=parent_uuid,
            content=[
                ToolResultItem(
                    status="accepted",
                    tool_call_id=tool_call_id,
                    content=None,
                )
            ],
        )
        return request.model_dump()

    async def _request_exact_tool_call(
        self,
        conversation_id: str,
        tool_name: str,
        requested_arguments: Dict[str, Any],
        parent_uuid: Optional[str] = None,
    ) -> tuple[str, str]:
        resolved_parent = await self._ensure_assistant_parent_uuid(
            conversation_id, parent_uuid=parent_uuid
        )
        post_result_instruction = ""
        if tool_name == "mcp__syntax-checker__validate":
            post_result_instruction = (
                "\nПосле результата инструмента ОБЯЗАТЕЛЬНО верни только итог проверки в простом тексте.\n"
                "Не возвращай JSON, markdown-блоки, пояснения о своих действиях и служебные комментарии.\n"
                "Если есть ошибки, перечисли их кратко списком.\n"
                "Если ошибок нет, напиши: \"Синтаксических ошибок не найдено\".\n"
            )
        elif tool_name == "mcp__knowledge-hub__Search_Documentation":
            post_result_instruction = (
                "\nПосле результата инструмента ОБЯЗАТЕЛЬНО верни обычным текстом содержательный ответ по найденной информации.\n"
                "Не ограничивайся сообщением о количестве найденных документов и их списком.\n"
                "Сформулируй полезную сводку по теме пользовательского запроса на основе найденных результатов.\n"
                "Если в найденной документации описаны процедуры или функции, по возможности укажи их входные параметры и возвращаемые значения.\n"
            )
        elif tool_name == "mcp__knowledge-hub__Search_ITS":
            post_result_instruction = (
                "\nПосле результата инструмента ОБЯЗАТЕЛЬНО верни обычным текстом содержательный развернутый ответ по теме запроса.\n"
                "Не ограничивайся сообщением о количестве найденных документов, их списком или коротким 'Выполнено'.\n"
                "Сформулируй полезную сводку по теме пользовательского запроса на основе найденных материалов.\n"
                "ОБЯЗАТЕЛЬНО укажи id вида its-*** ВСЕХ найденных документов, чтобы их можно было использовать в fetch_its.\n"
                "Не опускай и не изменяй идентификаторы вида its-***.\n"
            )
        elif tool_name == "mcp__knowledge-hub__Fetch_ITS":
            post_result_instruction = (
                "\nПосле результата инструмента ОБЯЗАТЕЛЬНО верни обычным текстом содержательный ПОДРОБНЫЙ развернутый ответ.\n"
                "Не оставляй ответ пустым и не ограничивайся коротким 'Выполнено'.\n"
                "Опиши, что найдено в документе, каталоге или базе, и приведи полезные ключевые сведения из результата.\n"
            )
        hidden_instruction = (
            "Внутренняя инструкция.\n"
            f"Нужно вернуть ровно один tool call для {tool_name}.\n"
            "Не используй другие инструменты.\n"
            "Сохрани все символы в аргументах без изменений.\n"
            f"Используй ровно эти JSON-аргументы: {json.dumps(requested_arguments, ensure_ascii=False)}\n"
        )
        hidden_instruction += (
            "Не отвечай обычным текстом до tool call.\n"
            f"{post_result_instruction}"
        )
        result = await self._send_user_message(
            conversation_id, hidden_instruction, parent_uuid=resolved_parent
        )
        assistant_uuid = result.get("assistant_uuid")
        tool_calls = result.get("tool_calls") or []
        if not assistant_uuid or not tool_calls:
            raise ApiError(f"Unable to obtain upstream tool call for {tool_name}")

        first_call = tool_calls[0]
        seen_name = (
            first_call.get("function", {}).get("name")
            if isinstance(first_call, dict)
            else None
        )
        tool_call_id = first_call.get("id") if isinstance(first_call, dict) else None
        if seen_name != tool_name or not tool_call_id:
            raise ApiError(f"Unexpected upstream tool call for {tool_name}")

        return assistant_uuid, tool_call_id

    @staticmethod
    def _parse_tool_arguments(raw_arguments: Any) -> Dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str) and raw_arguments:
            parsed = json.loads(raw_arguments)
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        return _RE_REASONING_BLOCK.sub("", text or "").strip()

    @staticmethod
    def _has_tool_name(tool_calls: list[dict[str, Any]], tool_name: str) -> bool:
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            if function.get("name") == tool_name:
                return True
        return False

    @staticmethod
    def _is_trivial_completion_text(text: str) -> bool:
        normalized = (text or "").strip().lower().rstrip(".!")
        if not normalized:
            return False
        return normalized in {"готово", "выполнено", "done", "ok", "ок"}

    def _prefer_final_text(
        self,
        current_text: str,
        candidate_text: str,
        *,
        todo_seen: bool,
    ) -> str:
        candidate = (candidate_text or "").strip()
        current = (current_text or "").strip()
        if not candidate:
            return current
        if not current:
            return candidate
        if todo_seen and self._is_trivial_completion_text(candidate):
            return current
        if self._is_trivial_completion_text(current) and not self._is_trivial_completion_text(candidate):
            return candidate
        return candidate

    @staticmethod
    def _merge_full_text(current_text: str, candidate_text: str) -> str:
        current = (current_text or "").strip()
        candidate = (candidate_text or "").strip()
        if not candidate:
            return current
        if not current:
            return candidate
        if candidate == current:
            return current
        if candidate in current:
            return current
        if current in candidate:
            return candidate
        if candidate.startswith(current):
            return candidate
        if current.startswith(candidate):
            return current
        max_overlap = min(len(current), len(candidate))
        for overlap in range(max_overlap, 0, -1):
            if current[-overlap:] == candidate[:overlap]:
                return f"{current}{candidate[overlap:]}"
        return f"{current}\n\n{candidate}"

    def _extract_task_result_text(self, tool_calls: list[dict[str, Any]]) -> Optional[str]:
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            if function.get("name") != "TaskResult":
                continue
            arguments = self._parse_tool_arguments(function.get("arguments"))
            result_text = arguments.get("result")
            if isinstance(result_text, str) and result_text.strip():
                return self._strip_reasoning(result_text)
        return None

    async def _respond_to_tool_call(
        self,
        conversation_id: str,
        *,
        assistant_uuid: str,
        tool_call: Dict[str, Any],
        ensure_ascii: bool = False,
    ) -> Dict[str, Any]:
        function = tool_call.get("function") or {}
        tool_name = function.get("name") or ""
        tool_call_id = tool_call.get("id") or ""
        if not tool_name or not tool_call_id:
            raise ApiError("Upstream tool call is missing tool name or id")
        payload = self._build_tool_ack_payload(
            parent_uuid=assistant_uuid,
            tool_call_id=tool_call_id,
        )
        return await self._collect_stream(conversation_id, payload)

    async def _collect_stream(
        self, conversation_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Ретраим только установку соединения. ConnectTimeout возникает до приёма
        # ответа, поэтому повтор всего стрима безопасен; ошибки в середине стрима
        # (после получения данных) прокидываются без повтора.
        return await self._retry_on_connect(
            lambda: self._collect_stream_once(conversation_id, payload),
            what=f"stream(conv={conversation_id})",
        )

    async def _collect_stream_once(
        self, conversation_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "assistant_uuid": None,
            "final_text": "",
            "full_text": "",
            "tool_calls": [],
            "tool_results": [],
            "tool_followups": [],
        }
        url = f"{self.base_url}/chat_api/v1/conversations/{conversation_id}/messages"
        request_headers = {"Accept": "text/event-stream"}
        accumulated_text = ""
        last_tool_call_id = ""
        assistant_segments: list[str] = []
        current_assistant_uuid: Optional[str] = None

        def build_visible_text(current_round_text: str) -> str:
            prefix = "\n\n".join(assistant_segments).strip()
            current = (current_round_text or "").strip()
            if prefix and current:
                return f"{prefix}\n\n{current}"
            if prefix:
                return prefix
            if current:
                return current
            return ""

        def append_assistant_segment(segment_text: str) -> str:
            segment = (segment_text or "").strip()
            if not segment:
                return build_visible_text("")
            if assistant_segments and assistant_segments[-1] == segment:
                return build_visible_text("")
            assistant_segments.append(segment)
            return build_visible_text("")

        logger.debug(
            "MCP upstream POST: url=%s conversation_id=%s payload=%s",
            url,
            conversation_id,
            payload,
        )

        async with self.client.stream(
            "POST", url, json=payload, headers=request_headers
        ) as response:
            logger.debug(
                "MCP upstream stream response: conversation_id=%s status=%s x-request-id=%s",
                conversation_id,
                response.status_code,
                response.headers.get("X-Request-ID"),
            )
            if response.status_code != 200:
                await response.aread()
                raise ApiError(
                    f"Message send error: {response.status_code}",
                    response.status_code,
                    data=self._extract_upstream_error_data(response),
                )

            response.encoding = "utf-8"
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    chunk = MessageChunk(**json.loads(line[6:]))
                except Exception as e:
                    logger.warning("MCP upstream parse/model error: %s", e)
                    continue

                if chunk.role == "assistant" and chunk.uuid:
                    current_assistant_uuid = chunk.uuid

                if chunk.role == "tool" and chunk.finished:
                    for ri in (chunk.render_info or []):
                        if not isinstance(ri, dict):
                            continue
                        result["tool_results"].append(
                            {
                                "tool_call_id": ri.get("tool_call_id", ""),
                                "tool_name": ri.get("tool_name", ""),
                                "response_markdown": ri.get("response_markdown") or "",
                                "response_details": (ri.get("details") or {}).get("response_details") or [],
                                "hide_after": ri.get("hide_after", True),
                            }
                        )
                    continue

                tc_from_content = (chunk.content or {}).get("tool_calls") if chunk.content else None
                if tc_from_content:
                    result["tool_calls"] = tc_from_content
                    last_tool_call_id = tc_from_content[-1].get("id", "") or last_tool_call_id

                raw_text = ""
                if chunk.content and chunk.content.get("content") is not None:
                    raw_text = chunk.content["content"]
                    accumulated_text = raw_text
                elif chunk.content_delta and chunk.content_delta.content is not None:
                    raw_text = chunk.content_delta.content
                    accumulated_text += raw_text

                current_visible_text = build_visible_text(accumulated_text)
                if current_visible_text:
                    result["full_text"] = current_visible_text

                if chunk.finished and chunk.role == "assistant":
                    assistant_uuid = chunk.uuid or current_assistant_uuid
                    result["assistant_uuid"] = assistant_uuid
                    result["final_text"] = (accumulated_text or raw_text or "").strip()
                    if assistant_uuid:
                        self._remember_assistant_uuid(conversation_id, assistant_uuid)

                    tc_list = (chunk.content or {}).get("tool_calls") or []
                    if tc_list:
                        result["tool_calls"] = tc_list
                        last_tool_call_id = tc_list[-1].get("id", "") or last_tool_call_id
                        result["full_text"] = append_assistant_segment(accumulated_text)
                        accumulated_text = ""
                    else:
                        result["full_text"] = build_visible_text(accumulated_text)
                    break

            fallback_visible_text = build_visible_text(accumulated_text)
            if fallback_visible_text:
                result["full_text"] = fallback_visible_text
                if not result["final_text"]:
                    result["final_text"] = fallback_visible_text

            if not result["assistant_uuid"] and current_assistant_uuid:
                result["assistant_uuid"] = current_assistant_uuid
                self._remember_assistant_uuid(conversation_id, current_assistant_uuid)

            followup_text = (result.get("full_text") or result.get("final_text") or "").strip()
            if last_tool_call_id and followup_text and result["tool_results"]:
                result["tool_followups"].append(
                    {
                        "tool_call_id": last_tool_call_id,
                        "text": followup_text,
                    }
                )

        return result

    async def call_task(
        self,
        conversation_id: str,
        *,
        instruction: str,
        skill: str,
        parent_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ = skill
        result = await self.call_prompt(
            conversation_id,
            instruction=instruction,
            parent_uuid=parent_uuid,
        )
        task_result_text = self._extract_task_result_text(result.get("tool_calls") or [])
        if task_result_text:
            result["final_text"] = task_result_text
            result["full_text"] = self._merge_full_text(
                result.get("full_text") or "",
                task_result_text,
            )
            result["tool_calls"] = []
        return result

    async def call_exact_tool(
        self,
        conversation_id: str,
        *,
        tool_name: str,
        arguments: Dict[str, Any],
        parent_uuid: Optional[str] = None,
        payload_ensure_ascii: bool = False,
    ) -> Dict[str, Any]:
        assistant_uuid, tool_call_id = await self._request_exact_tool_call(
            conversation_id,
            tool_name,
            arguments,
            parent_uuid=parent_uuid,
        )
        _ = payload_ensure_ascii
        payload = self._build_tool_ack_payload(
            parent_uuid=assistant_uuid,
            tool_call_id=tool_call_id,
        )
        return await self._collect_stream(conversation_id, payload)
