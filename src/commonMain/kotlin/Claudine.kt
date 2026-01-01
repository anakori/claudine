/*
 * claudine - an autonomous and Unix-omnipotent AI agent using Anthropic API
 * Copyright (C) 2025 Kazimierz Pogoda / Xemantic
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package com.xemantic.ai.claudine

import com.xemantic.ai.anthropic.Anthropic
import com.xemantic.ai.anthropic.AnthropicConfigException
import com.xemantic.ai.anthropic.cache.CacheControl
import com.xemantic.ai.anthropic.cost.CostWithUsage
import com.xemantic.ai.anthropic.cost.costReport
import com.xemantic.ai.anthropic.message.Message
import com.xemantic.ai.anthropic.message.StopReason
import com.xemantic.ai.anthropic.message.System
import com.xemantic.ai.anthropic.message.addCacheBreakpoint
import com.xemantic.ai.anthropic.message.plusAssign
import com.xemantic.ai.anthropic.tool.Toolbox
import com.xemantic.ai.claudine.tool.CreateFile
import com.xemantic.ai.claudine.tool.ExecuteShellCommand
import com.xemantic.ai.claudine.tool.OpenUrl
import com.xemantic.ai.claudine.tool.ReadBinaryFiles
import com.xemantic.ai.claudine.tool.ReadFiles
import com.xemantic.ai.claudine.tool.describeTools
import io.ktor.client.HttpClient

/**
 * Minimum token count required for prompt caching to be effective.
 * Below this threshold, caching costs more than it saves.
 */
private const val MIN_CACHEABLE_TOKENS = 1024

/**
 * Adds a cache breakpoint to the last message in the conversation with the specified TTL,
 * but only if the conversation is large enough to benefit from caching.
 */
private fun List<Message>.addCacheBreakpointIfWorthwhile(
    conversationTokens: Int,
    ttl: CacheControl.Ephemeral.TTL = CacheControl.Ephemeral.TTL.ONE_HOUR
): List<Message> {
    return if (conversationTokens >= MIN_CACHEABLE_TOKENS) {
        mapLast { message ->
            message.copy {
                content = content.mapLast { contentElement ->
                    contentElement.alterCacheControl(
                        CacheControl.Ephemeral { this.ttl = ttl }
                    )
                }
            }
        }
    } else {
        this // Don't add cache breakpoint if conversation is too small
    }
}

/**
 * Helper extension to transform the last element of a list.
 */
private inline fun <T> List<T>.mapLast(transform: (T) -> T): List<T> {
    return if (isEmpty()) this
    else dropLast(1) + transform(last())
}

val claudineSystemPrompt = """
Your name is Claudine and you are an AI agent controlling the machine of the human you are connected to while using cognition of the Claude AI LLM model.

You are provided with tools to fulfill this purpose.

IMPORTANT: Always check file sizes before reading or processing them, especially for images and other potentially large files.

When reading files:
- First, use the ExecuteShellCommand tool to check the file size (e.g., `ls -l <filename>` or `stat -f%z <filename>`).
- For image and document formats supported by Claude models use ReadBinaryFiles tool.
- For image files larger than 5 KB:
   a. Use local image conversion tools to create a miniature version.
   b. The miniature should keep the aspect ratio of the original image and have a width of 512 pixels (e.g., `convert input.jpg -resize 512x temp_output.webp`).
   c. Use an image format with higher compression (e.g., WebP or JPEG) for the miniature.
   d. Store temporary files in the system's default temporary folder with unique names.

File operations:
- Prefer using shell commands (via ExecuteShellCommand) for copying or moving files instead of ReadFiles and CreateFile tools.
- When listing files with ExecuteShellCommand, use recursive lists with a maximum depth of 2 levels, to minimize the amount of excessive information quickly filling up the token window.

When analyzing the source code, work in 2 steps:
1. First establish the complete list of all the project source files using ExecuteShellCommand tool.
2. Then read all the text files at once using ReadFiles tool.

Always verify file sizes and types before processing, and never assume a file is small enough to read directly without checking first.

Your source code is located at:

git@github.com:xemantic/claudine.git

The operating system of human's machine: $operatingSystem

"""

/**
 * Starts claudine agent.
 *
 * @return the exit code
 */
suspend fun claudine(): Int {

    val httpClient = HttpClient()

    val anthropic = try {
        Anthropic {
            anthropicBeta = listOf(
                "token-efficient-tools-2025-02-19",
                "prompt-caching-2024-07-31"
            )
        }
    } catch (e: AnthropicConfigException) {
        println(e.message)
        return 1 // exit code
    }

    println("[Claudine]> Connecting human and human's machine to cognition of Claude AI")

    var totalStats = CostWithUsage.ZERO
    val conversation = mutableListOf<Message>()
    var conversationTokens = 0 // Track conversation size for conditional caching

    val systemPrompt = listOf(
        System(
            text = """
                $claudineSystemPrompt

                ${describeCurrentMoment()}
            """.trimIndent(),
            cacheControl = CacheControl.Ephemeral {
                ttl = CacheControl.Ephemeral.TTL.ONE_HOUR
            }
        )
    )

    val toolbox = Toolbox {
        tool<ExecuteShellCommand> { use() }
        tool<CreateFile> { use() }
        tool<ReadBinaryFiles> { use() }
        tool<ReadFiles> { use() }
        tool<OpenUrl> { use(httpClient) }
    }

    while (true) {

        print("[me]> ")

        val input = readln()
        if (input == "exit") break

        conversation += input

        println("[Claudine] ...Reasoning...")

        var agentLoop = false
        do {
            if (agentLoop) {
                println("[Claudine] ...Processing tool results...")
            }
            val response = anthropic.messages.create {
                system = systemPrompt
                messages = conversation.addCacheBreakpointIfWorthwhile(conversationTokens)
                tools = toolbox.tools
            }
            if (response.stopReason == StopReason.MAX_TOKENS) {
                println("[Claudine]> Error: max number of output tokens reached")
                conversation[conversation.lastIndex] = conversation.last().copy {
                    +"Limit the output not to exceed the limit of 64000 tokens"
                }
                continue
            }

            conversation += response

            // Update conversation token count for next turn
            // Accumulate tokens to track total conversation size
            // inputTokens represents non-cached input on this turn
            conversationTokens += response.usage.inputTokens

            response.text?.run {
                println("[Claudine]> ${response.text}")
            }

            val stats = response.costWithUsage
            totalStats += stats

            println("[Claudine]> Tax:")
            print(costReport(stats, totalStats))
            println("|")

            agentLoop = if (response.stopReason == StopReason.TOOL_USE) {
                response.describeTools()
                conversation += response.useTools(toolbox)
                true
            } else {
                false
            }

        } while (agentLoop)

    }

    return 0 // exit code
}
