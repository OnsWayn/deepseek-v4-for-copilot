import type { CancellationToken } from 'vscode';
import { safeStringify } from '../json';
import { logger } from '../logger';
import type {
	DeepSeekRequest,
	DeepSeekStreamChunk,
	DeepSeekToolCall,
	DeepSeekUsage,
	StreamCallbacks,
} from '../types';
import { createHttpError, formatRequestError, normalizeRequestError, DeepSeekRequestError } from './error';

/**
 * Lightweight SSE-streaming DeepSeek API client.
 * Supports standard /chat/completions as well as Responses API (/responses with web_search).
 * No external dependencies — uses Node's built-in fetch.
 */
export class DeepSeekClient {
	private readonly cleanBaseUrl: string;

	constructor(
		private readonly baseUrl: string,
		private readonly apiKey: string,
	) {
		this.cleanBaseUrl = baseUrl.replace(/\/+$/u, '');
	}

	/**
	 * Stream a chat completion or response from the API.
	 * Parses SSE chunks and dispatches callbacks for content, thinking, and tool calls.
	 */
	async streamChatCompletion(
		request: DeepSeekRequest,
		callbacks: StreamCallbacks,
		cancellationToken?: CancellationToken,
	): Promise<void> {
		const controller = new AbortController();
		const cancelListener = cancellationToken?.onCancellationRequested(() => {
			controller.abort();
		});
		if (cancellationToken?.isCancellationRequested) {
			controller.abort();
		}

		try {
			let response: Response | undefined;

			if (request.webSearch) {
				// Strict Responses API tools format
				const tools: Array<{ type: string; name?: string; description?: string; parameters?: Record<string, unknown> }> = [
					{ type: 'web_search' },
				];
				if (request.tools && request.tools.length > 0) {
					for (const t of request.tools) {
						if (t.type === 'function' && t.function) {
							tools.push({
								type: 'function',
								name: t.function.name,
								description: t.function.description,
								parameters: t.function.parameters,
							});
						}
					}
				}

				const inputItems = (request.messages || []).map((m) => ({
					role: m.role,
					content: m.content || '',
				}));

				const responsesPayload: Record<string, unknown> = {
					model: request.model,
					input: inputItems,
					tools,
					stream: true,
					...(request.max_tokens ? { max_output_tokens: request.max_tokens } : {}),
					...(request.reasoning_effort && request.reasoning_effort !== 'none'
						? { reasoning: { effort: request.reasoning_effort } }
						: {}),
				};

				const candidateUrls = [
					`${this.cleanBaseUrl}/responses`,
					`${this.cleanBaseUrl}/v1/responses`,
				];

				for (const targetUrl of candidateUrls) {
					try {
						const res = await fetch(targetUrl, {
							method: 'POST',
							headers: {
								'Content-Type': 'application/json',
								Authorization: `Bearer ${this.apiKey}`,
							},
							body: safeStringify(responsesPayload),
							signal: controller.signal,
						});

						if (res.ok) {
							response = res;
							break;
						} else if (res.status === 404) {
							continue;
						} else {
							throw await createHttpError(res, { baseUrl: this.baseUrl, request });
						}
					} catch (err) {
						if (isAbortError(err) && cancellationToken?.isCancellationRequested) {
							return;
						}
						if (err instanceof DeepSeekRequestError) {
							throw err;
						}
						logger.warn(`Failed to connect to Responses API candidate ${targetUrl}:`, err);
					}
				}
			}

			// Fallback or standard route: /chat/completions
			if (!response) {
				const validTools = request.tools?.filter((t) => t.type === 'function');
				const chatBody = {
					model: request.model,
					messages: request.messages,
					stream: true,
					tools: validTools && validTools.length > 0 ? validTools : undefined,
					tool_choice: validTools && validTools.length > 0 ? request.tool_choice || 'auto' : undefined,
					max_tokens: request.max_tokens,
					stream_options: { include_usage: true },
					...(request.thinking ? { thinking: request.thinking } : {}),
					...(request.reasoning_effort ? { reasoning_effort: request.reasoning_effort } : {}),
				};

				const chatUrl = `${this.cleanBaseUrl}/chat/completions`;
				response = await fetch(chatUrl, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						Authorization: `Bearer ${this.apiKey}`,
					},
					body: safeStringify(chatBody),
					signal: controller.signal,
				});

				if (!response.ok) {
					throw await createHttpError(response, { baseUrl: this.baseUrl, request });
				}
			}

			if (!response.body) {
				throw new Error('No response body received');
			}

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';
			let latestUsage: DeepSeekUsage | undefined;

			// Accumulate tool call deltas by index or string id
			const pendingToolCalls = new Map<string | number, DeepSeekToolCall>();

			while (true) {
				if (cancellationToken?.isCancellationRequested) {
					controller.abort();
					return;
				}

				const { done, value } = await reader.read();
				if (done) {
					break;
				}

				buffer += decoder.decode(value, { stream: true });

				const lines = buffer.split('\n');
				buffer = lines.pop() || '';

				for (const line of lines) {
					const trimmed = line.trim();

					if (!trimmed || trimmed.startsWith(':')) {
						continue;
					}

					if (trimmed === 'data: [DONE]') {
						for (const tc of pendingToolCalls.values()) {
							callbacks.onToolCall(tc);
						}
						pendingToolCalls.clear();
						reportFinalUsage(callbacks, latestUsage);
						callbacks.onDone();
						return;
					}

					if (!trimmed.startsWith('data: ')) {
						continue;
					}

					const jsonStr = trimmed.slice(6);
					try {
						const chunk = JSON.parse(jsonStr);
						if (chunk.error) {
							throw new Error(chunk.error.message || JSON.stringify(chunk.error));
						}

						if (chunk.usage) {
							latestUsage = chunk.usage;
						}

						// 1. Responses API semantic events
						if (chunk.type) {
							const type = String(chunk.type);
							if (type === 'response.output_text.delta' && (chunk.delta || chunk.text)) {
								callbacks.onContent(chunk.delta || chunk.text || '');
							} else if (
								(type === 'response.reasoning_text.delta' ||
									type === 'response.reasoning_content.delta') &&
								(chunk.delta || chunk.text)
							) {
								callbacks.onThinking(chunk.delta || chunk.text || '');
							} else if (
								type.startsWith('response.web_search_call') ||
								(type === 'response.output_item.added' && chunk.item?.type === 'web_search_call')
							) {
								const q = chunk.query || chunk.item?.action?.query || chunk.item?.query || '';
								if (
									type === 'response.web_search_call.searching' ||
									type === 'response.web_search_call.in_progress' ||
									type === 'response.output_item.added'
								) {
									callbacks.onThinking(`\n🔍 [联网搜索] 正在检索${q ? ` "${q}"` : ''}...\n`);
								} else if (
									type === 'response.web_search_call.completed' ||
									(type === 'response.output_item.done' && chunk.item?.type === 'web_search_call')
								) {
									callbacks.onThinking(`\n✅ [联网搜索] 检索完成\n`);
								}
							} else if (type === 'response.function_call_arguments.delta') {
								const id = chunk.call_id || chunk.item_id || 'call_0';
								let pending = pendingToolCalls.get(id);
								if (!pending) {
									pending = {
										id,
										type: 'function',
										function: { name: chunk.name || '', arguments: '' },
									};
									pendingToolCalls.set(id, pending);
								}
								if (chunk.delta) {
									pending.function.arguments += chunk.delta;
								}
							} else if (type === 'response.output_item.done' && chunk.item?.type === 'function_call') {
								const item = chunk.item;
								const id = item.id || item.call_id || 'call_0';
								const args =
									typeof item.arguments === 'string'
										? item.arguments
										: JSON.stringify(item.arguments || {});
								callbacks.onToolCall({
									id,
									type: 'function',
									function: { name: item.name || '', arguments: args },
								});
								pendingToolCalls.delete(id);
							} else if (type === 'response.completed') {
								if (chunk.response?.usage) {
									latestUsage = chunk.response.usage;
								}
								for (const tc of pendingToolCalls.values()) {
									callbacks.onToolCall(tc);
								}
								pendingToolCalls.clear();
								reportFinalUsage(callbacks, latestUsage);
								callbacks.onDone();
								return;
							}
						}

						// 2. Chat Completions choices format
						const choice = (chunk as DeepSeekStreamChunk).choices?.[0];
						if (choice) {
							const reasoning = choice.delta?.reasoning_content;
							if (reasoning) {
								callbacks.onThinking(reasoning);
							}

							if (choice.delta?.content) {
								callbacks.onContent(choice.delta.content);
							}

							if (choice.delta?.tool_calls) {
								for (const tc of choice.delta.tool_calls) {
									let pending = pendingToolCalls.get(tc.index);
									if (!pending && tc.id) {
										pending = {
											id: tc.id,
											type: 'function',
											function: { name: '', arguments: '' },
										};
										pendingToolCalls.set(tc.index, pending);
									}
									if (pending) {
										if (tc.function?.name) {
											pending.function.name += tc.function.name;
										}
										if (tc.function?.arguments) {
											pending.function.arguments += tc.function.arguments;
										}
									}
								}
							}

							if (choice.finish_reason === 'tool_calls' || choice.finish_reason === 'stop') {
								for (const tc of pendingToolCalls.values()) {
									callbacks.onToolCall(tc);
								}
								pendingToolCalls.clear();
							}
						}
					} catch (e) {
						logger.error('Failed to parse SSE chunk:', jsonStr.slice(0, 200), e);
					}
				}
			}

			reportFinalUsage(callbacks, latestUsage);
			callbacks.onDone();
		} catch (error) {
			if (isAbortError(error) && cancellationToken?.isCancellationRequested) {
				return;
			}
			const normalizedError = normalizeRequestError(error, { baseUrl: this.baseUrl, request });
			logger.error('DeepSeek request failed:', formatRequestError(normalizedError));
			callbacks.onError(normalizedError);
		} finally {
			cancelListener?.dispose();
		}
	}
}

function reportFinalUsage(callbacks: StreamCallbacks, usage: DeepSeekUsage | undefined): void {
	if (!usage || !callbacks.onUsage) {
		return;
	}
	callbacks.onUsage(usage);
}

function isAbortError(error: unknown): boolean {
	return error instanceof Error && error.name === 'AbortError';
}
