#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// Qwen's tool syntax carries each argument as untyped text. This terminal contract records only
// the supported top-level JSON Schema types needed to normalize that text. A type mismatch remains
// a structured call for consumer validation; recursive validation is outside this non-strict
// contract.
struct ToolCallOutputContract {
    enum class SchemaType : std::uint8_t {
        Null    = 1U << 0U,
        Boolean = 1U << 1U,
        Integer = 1U << 2U,
        Number  = 1U << 3U,
        String  = 1U << 4U,
        Object  = 1U << 5U,
        Array   = 1U << 6U,
    };

    struct TypeSet {
        std::uint8_t bits = 0;
    };

    enum class NormalizationPolicy : std::uint8_t {
        Legacy,
        DeclaredTypes,
    };

    struct Parameter {
        std::string name;
        NormalizationPolicy policy = NormalizationPolicy::Legacy;
        TypeSet types;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        bool unambiguous = true;
    };

    std::vector<Tool> tools;
    bool enforce_declared_names = false;
};

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled);

[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract);

// One feed() result: text provably outside a tool-call suffix, followed by the tool-call
// fragments this chunk committed (in order).
struct FeedResult {
    std::string content;
    std::vector<ToolCallStreamFragment> fragments;
};

// Incrementally publishes bytes that are provably outside a possible terminal Qwen tool-call
// suffix, plus tool-call fragments as the region structure commits. At terminal time, valid calls
// are retained structurally; a structurally incomplete region is finalized from the committed
// fragments. A region that fails before any fragment committed is restored verbatim as content.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
        // Fragments emitted while finalizing an open call at terminal (e.g. the closing argument
        // brace of a call that never closed). These must be published before the terminal events.
        std::vector<ToolCallStreamFragment> fragments;
    };

    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length);

    // Publishes text provably outside a tool-call suffix, plus fragments committed by this chunk.
    // Fragments always follow the visible content of the same call; concatenating the Arguments
    // fragments of each call reproduces its terminal arguments JSON.
    [[nodiscard]] FeedResult feed(std::string_view text);
    [[nodiscard]] Terminal finish();

private:
    // Incremental region phases; see the decoder implementation for their transitions.
    enum class RegionPhase : std::uint8_t {
        Content,             // outside any tool region: scanning for the open marker
        ContentPassthrough,  // region failed before any fragment: all bytes are content
        ExpectToolCall,      // expecting <tool_call>
        FunctionHeader,      // expecting <function=name>
        FunctionBody,        // expecting </function> or <parameter=
        ParameterHeader,     // expecting <parameter=name>
        ParameterValue,      // expecting </parameter> (nested opens raise depth)
        ExpectToolCallClose, // expecting </tool_call>
        Discard,             // region failed after fragments: committed calls stand, tail dropped
    };

    void parse_region(FeedResult& result);
    bool parse_expect_tool_call(FeedResult& result);
    bool parse_function_header(FeedResult& result);
    bool parse_function_body(FeedResult& result);
    bool parse_expect_tool_call_close(FeedResult& result);
    bool parse_parameter_header(FeedResult& result);
    bool parse_parameter_value(FeedResult& result);
    void fail(FeedResult& result, ToolCallParseFallbackReason reason);
    void flush_region_if_failed(FeedResult& result);

    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string trailing_whitespace_;
    std::string region_;
    std::size_t marker_prefix_bytes_  = 0;
    std::size_t max_tool_name_length_ = 0;
    bool finished_                    = false;

    RegionPhase phase_ = RegionPhase::Content;
    std::size_t pos_   = 0;
    bool call_open_    = false;
    std::string call_name_;
    std::string call_arguments_;
    std::vector<std::string> call_param_names_;
    std::string param_name_;
    std::vector<GeneratedToolCall> committed_;
    std::uint32_t call_index_          = 0;
    bool any_fragment_emitted_         = false;
    bool failed_                       = false;
    bool pre_commit_failure_           = false;
    ToolCallParseDiagnostics diagnostics_;
};

} // namespace ninfer::models::qwen3_5::frontend
