#include "models/qwen3_5/program/planning/pressure_planner.h"

namespace ninfer::models::qwen3_5::detail {

namespace planning_detail {

inline constexpr std::size_t kOptionalTargetCapacity = 262144;

void hash_mix(std::uint64_t& hash, std::uint64_t value) noexcept {
    hash ^= value;
    hash *= 1099511628211ULL;
}

[[nodiscard]] std::uint64_t target_hash(std::uint32_t candidate_index,
                                        std::span<const std::uint16_t> choices) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    hash_mix(hash, candidate_index);
    for (const std::uint16_t choice : choices) { hash_mix(hash, choice); }
    return hash;
}

[[nodiscard]] bool same_choices(std::span<const std::uint16_t> left,
                                std::span<const std::uint16_t> right) noexcept {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

} // namespace planning_detail

PressurePlanningSessionImpl::PressurePlanningSessionImpl(
    Core& owner, std::span<const PhysicalCandidateBinding> physical_candidates,
    std::span<const runtime::PlanningCandidateId> admission_candidate_ids,
    std::span<const ContinuationHandle* const> private_owners,
    std::span<const runtime::PlanningOwnerId> private_owner_ids,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const runtime::PlanningOwnerId> shared_owner_ids)
    : program(&owner), resource_revision(owner.resource_revision()) {
    if (physical_candidates.empty() ||
        physical_candidates.size() != admission_candidate_ids.size() ||
        private_owners.size() != private_owner_ids.size() ||
        shared_owners.size() != shared_owner_ids.size() || owner.has_context_transaction() ||
        owner.pending_transaction_ || owner.pressure_planning_active_) {
        throw std::logic_error("pressure planning session cannot start in the current state");
    }

    candidates.assign(physical_candidates.begin(), physical_candidates.end());
    candidate_ids.assign(admission_candidate_ids.begin(), admission_candidate_ids.end());
    for (std::size_t index = 0; index < candidate_ids.size(); ++index) {
        if (std::find(candidate_ids.begin(), candidate_ids.begin() + index, candidate_ids[index]) !=
            candidate_ids.begin() + index) {
            throw std::logic_error("pressure planning candidate ID is duplicated");
        }
    }
    owners.reserve(private_owners.size() + shared_owners.size());
    for (std::size_t index = 0; index < private_owners.size(); ++index) {
        const ContinuationHandle* handle = private_owners[index];
        if (handle == nullptr || !owner.valid_continuation(*handle)) {
            throw std::logic_error("pressure planning private owner is stale");
        }
        owners.push_back(
            Owner{.private_handle = handle, .id = private_owner_ids[index], .shared = false});
    }
    for (std::size_t index = 0; index < shared_owners.size(); ++index) {
        const SharedPrefixHandle* handle = shared_owners[index];
        if (handle == nullptr || !owner.valid_shared_prefix(*handle)) {
            throw std::logic_error("pressure planning shared owner is stale");
        }
        owners.push_back(
            Owner{.shared_handle = handle, .id = shared_owner_ids[index], .shared = true});
    }
    std::sort(owners.begin(), owners.end(), [](const Owner& left, const Owner& right) {
        return std::tuple{left.id.value, left.shared} < std::tuple{right.id.value, right.shared};
    });
    for (std::size_t index = 1; index < owners.size(); ++index) {
        if (owners[index - 1].id == owners[index].id) {
            throw std::logic_error("pressure planning owner ID is duplicated");
        }
    }

    candidate_options.resize(candidates.size());
    const std::size_t maximum_targets =
        candidates.size() + 1U + planning_detail::kOptionalTargetCapacity;
    const std::size_t maximum_successors_per_owner =
        11U + owner.context_cache.max_long_anchors_per_continuation.value_or(0);
    if (!owners.empty() &&
        maximum_successors_per_owner > std::numeric_limits<std::size_t>::max() / owners.size()) {
        throw std::overflow_error("pressure expansion arena size overflow");
    }
    const std::size_t maximum_scratch_targets = owners.size() * maximum_successors_per_owner;
    targets.reserve(maximum_targets);
    if (maximum_scratch_targets > std::numeric_limits<std::size_t>::max() - maximum_targets ||
        (!owners.empty() && maximum_targets + maximum_scratch_targets >
                                std::numeric_limits<std::size_t>::max() / owners.size())) {
        throw std::overflow_error("pressure target choice arena size overflow");
    }
    target_choice_arena.reserve((maximum_targets + maximum_scratch_targets) * owners.size());
    choice_scratch.reserve(owners.size());
    std::size_t hash_capacity = 1;
    while (hash_capacity < 2U * maximum_targets) { hash_capacity <<= 1U; }
    target_hash_table.assign(hash_capacity, std::numeric_limits<std::uint32_t>::max());
    expansion_scratch.reserve(maximum_scratch_targets);
    committed_children.reserve(maximum_scratch_targets);
    for (auto& cursor : construction_slots) {
        cursor.choices.reserve(owners.size());
        cursor.options.reserve(maximum_scratch_targets + owners.size());
    }
    guidance_recovery.reserve(owners.size());
    selected_private_owners.reserve(private_owners.size());
    selected_private_owner_ids.reserve(private_owners.size());
    selected_private_decisions.reserve(private_owners.size());
    selected_shared_owners.reserve(shared_owners.size());
    selected_shared_owner_ids.reserve(shared_owners.size());
    selected_shared_decisions.reserve(shared_owners.size());
    recovery_private_owners.reserve(private_owners.size());
    recovery_private_decisions.reserve(private_owners.size());
    recovery_private_owner_ids.reserve(private_owners.size());
    recovery_shared_owners.reserve(shared_owners.size());
    recovery_shared_decisions.reserve(shared_owners.size());
    recovery_shared_owner_ids.reserve(shared_owners.size());
    projected_owner_decisions.assign(owners.size(), nullptr);
    assessment_outcomes.reserve(owners.size());
    guidance_outcomes.reserve(owners.size());
    guidance_checkpoint_changes.reserve(
        owners.size() * (2U + owner.context_cache.max_long_anchors_per_continuation.value_or(0)));
    const std::size_t private_checkpoint_capacity =
        2U + owner.context_cache.max_long_anchors_per_continuation.value_or(0);
    if (private_checkpoint_capacity > std::numeric_limits<std::size_t>::max() - 3U ||
        private_checkpoint_capacity != 0 &&
            private_checkpoint_capacity >
                std::numeric_limits<std::size_t>::max() / (private_checkpoint_capacity + 3U)) {
        throw std::overflow_error("pressure recovery alternative capacity overflow");
    }
    const std::size_t alternatives_per_private =
        private_checkpoint_capacity * (private_checkpoint_capacity + 3U) / 2U;
    if ((!private_owners.empty() &&
         alternatives_per_private >
             std::numeric_limits<std::size_t>::max() / private_owners.size()) ||
        shared_owners.size() > (std::numeric_limits<std::size_t>::max() -
                                private_owners.size() * alternatives_per_private) /
                                   2U) {
        throw std::overflow_error("pressure recovery alternative arena overflow");
    }
    const std::size_t recovery_alternative_capacity =
        private_owners.size() * alternatives_per_private + shared_owners.size() * 2U;
    if ((!private_owners.empty() &&
         private_checkpoint_capacity >
             (std::numeric_limits<std::size_t>::max() - shared_owners.size()) /
                 private_owners.size())) {
        throw std::overflow_error("pressure recovery impact arena overflow");
    }
    const std::size_t recovery_impact_capacity =
        private_owners.size() * private_checkpoint_capacity + shared_owners.size();
    if (owners.size() > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::overflow_error("pressure recovery State scratch capacity overflow");
    }
    assessment_impact_projections.reserve(recovery_impact_capacity);
    assessment_recovery_alternatives.reserve(recovery_alternative_capacity);
    recovery_scratch.state_placements.reserve(3U * owners.size());
    recovery_scratch.owners.reserve(owners.size());
    recovery_scratch.checkpoints.reserve(private_checkpoint_capacity);
    recovery_scratch.direct_work.reserve(private_checkpoint_capacity);
    recovery_scratch.continuation_summary.long_anchors.reserve(
        owner.context_cache.max_long_anchors_per_continuation.value_or(0));
    for (AssessmentSlot& slot : assessment_slots) {
        slot.owner_outcomes.reserve(owners.size());
        slot.checkpoint_impacts.reserve(recovery_impact_capacity);
        slot.recovery_alternatives.reserve(recovery_alternative_capacity);
    }
    using PlanningContractAccess = qwen3_5::detail::RuntimeContractAccess;

    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const PhysicalCandidateBinding& binding = candidates[index];
        const bool admission                    = binding.admission != nullptr;
        const bool capture                      = binding.capture != nullptr;
        if (binding.state == nullptr || admission == capture ||
            (admission && binding.state != static_cast<const CandidateState*>(binding.admission)) ||
            (capture && binding.state != static_cast<const CandidateState*>(binding.capture)) ||
            binding.state->planning_revision != resource_revision) {
            throw std::logic_error("pressure planning candidate is stale");
        }
        CandidateOptions& options = candidate_options[index];
        options.victims.reserve(owners.size());
        for (std::size_t owner_index = 0; owner_index < owners.size(); ++owner_index) {
            const Owner& pressure_owner = owners[owner_index];
            const bool selected_private_source =
                !pressure_owner.shared && binding.state->has_source &&
                PlanningContractAccess::index(*pressure_owner.private_handle) ==
                    binding.state->source_index &&
                PlanningContractAccess::epoch(*pressure_owner.private_handle) ==
                    binding.state->source_generation;
            const bool selected_shared_source =
                pressure_owner.shared && binding.state->has_shared_source &&
                PlanningContractAccess::index(*pressure_owner.shared_handle) ==
                    binding.state->shared_source_index &&
                PlanningContractAccess::epoch(*pressure_owner.shared_handle) ==
                    binding.state->shared_source_generation;
            if (!selected_private_source && !selected_shared_source) {
                options.victims.push_back(CandidateVictimOptions{
                    .owner_index = static_cast<std::uint32_t>(owner_index),
                });
            }
        }
        choice_scratch.assign(options.victims.size(), 0);
        if (intern_target(static_cast<std::uint32_t>(index), choice_scratch) != index) {
            throw std::logic_error("pressure identity target ordinal changed");
        }
    }

    if (++owner.pressure_planning_generation_ == 0) { ++owner.pressure_planning_generation_; }
    generation                      = owner.pressure_planning_generation_;
    owner.pressure_planning_active_ = true;
}

PressurePlanningSessionImpl::~PressurePlanningSessionImpl() noexcept {
    if (std::any_of(assessment_slots.begin(), assessment_slots.end(),
                    [](const AssessmentSlot& slot) { return slot.leased; })) {
        std::terminate();
    }
    if (program != nullptr) { program->pressure_planning_active_ = false; }
}

std::uint32_t PressurePlanningSessionImpl::acquire_assessment_slot() {
    for (std::uint32_t index = 0; index < assessment_slots.size(); ++index) {
        AssessmentSlot& slot = assessment_slots[index];
        if (slot.leased) { continue; }
        slot.leased = true;
        slot.owner_outcomes.clear();
        slot.checkpoint_impacts.clear();
        slot.recovery_alternatives.clear();
        return index;
    }
    throw std::logic_error("pressure runner retained too many simultaneous assessments");
}

void PressurePlanningSessionImpl::release_assessment_slot(const void* owner,
                                                          std::uint32_t slot_index,
                                                          std::uint32_t slot_generation) noexcept {
    auto* session = const_cast<PressurePlanningSessionImpl*>(
        static_cast<const PressurePlanningSessionImpl*>(owner));
    if (session == nullptr || slot_index >= session->assessment_slots.size()) { std::terminate(); }
    AssessmentSlot& slot = session->assessment_slots[slot_index];
    if (!slot.leased || slot.generation != slot_generation) { std::terminate(); }
    slot.owner_outcomes.clear();
    slot.checkpoint_impacts.clear();
    slot.recovery_alternatives.clear();
    slot.leased = false;
    if (++slot.generation == 0) { ++slot.generation; }
}

bool PressurePlanningSessionImpl::valid(qwen3_5::PressureTargetHandle target) const noexcept {
    return target.session_ == this && target.generation_ == generation &&
           target.index_ < targets.size() && program != nullptr &&
           program->resource_revision() == resource_revision;
}

std::uint32_t
PressurePlanningSessionImpl::candidate_index(runtime::PlanningCandidateId candidate) const {
    const auto found = std::find(candidate_ids.begin(), candidate_ids.end(), candidate);
    if (found == candidate_ids.end()) {
        throw std::invalid_argument("pressure target candidate does not belong to this session");
    }
    return static_cast<std::uint32_t>(found - candidate_ids.begin());
}

const PressurePlanningSessionImpl::TargetNode*
PressurePlanningSessionImpl::find_target(std::uint32_t selected_candidate,
                                         std::span<const std::uint16_t> choices) const noexcept {
    if (target_hash_table.empty()) { return nullptr; }
    const std::size_t mask = target_hash_table.size() - 1U;
    std::size_t slot =
        static_cast<std::size_t>(planning_detail::target_hash(selected_candidate, choices)) & mask;
    for (std::size_t probe = 0; probe < target_hash_table.size(); ++probe) {
        const std::uint32_t index = target_hash_table[slot];
        if (index == std::numeric_limits<std::uint32_t>::max()) { return nullptr; }
        if (index < targets.size()) {
            const TargetNode& target = targets[index];
            if (target.candidate_index == selected_candidate &&
                target.victim_choice_offset <= target_choice_arena.size() &&
                target.victim_choice_count <=
                    target_choice_arena.size() - target.victim_choice_offset &&
                std::equal(choices.begin(), choices.end(),
                           target_choice_arena.begin() + target.victim_choice_offset,
                           target_choice_arena.begin() + target.victim_choice_offset +
                               target.victim_choice_count)) {
                return &target;
            }
        }
        slot = (slot + 1U) & mask;
    }
    return nullptr;
}

PressurePlanningSessionImpl::TargetNode*
PressurePlanningSessionImpl::find_target(std::uint32_t selected_candidate,
                                         std::span<const std::uint16_t> choices) noexcept {
    return const_cast<TargetNode*>(std::as_const(*this).find_target(selected_candidate, choices));
}

std::span<const std::uint16_t>
PressurePlanningSessionImpl::victim_choices(const TargetNode& target) const {
    if (target.victim_choice_offset > target_choice_arena.size() ||
        target.victim_choice_count > target_choice_arena.size() - target.victim_choice_offset) {
        throw std::logic_error("pressure target choice span is invalid");
    }
    return std::span<const std::uint16_t>(target_choice_arena)
        .subspan(target.victim_choice_offset, target.victim_choice_count);
}

std::uint32_t PressurePlanningSessionImpl::intern_target(std::uint32_t selected_candidate,
                                                         std::span<const std::uint16_t> choices,
                                                         bool root_maximal) {
    if (selected_candidate >= candidate_options.size() ||
        choices.size() != candidate_options[selected_candidate].victims.size()) {
        throw std::logic_error("pressure target does not match its candidate victim domain");
    }
    if (TargetNode* existing = find_target(selected_candidate, choices)) {
        existing->root_maximal = existing->root_maximal || root_maximal;
        return static_cast<std::uint32_t>(existing - targets.data());
    }
    const std::size_t maximum = candidates.size() + 1U + planning_detail::kOptionalTargetCapacity;
    if (targets.size() >= maximum || targets.size() == targets.capacity() ||
        choices.size() > target_choice_arena.capacity() - target_choice_arena.size() ||
        target_choice_arena.size() > std::numeric_limits<std::uint32_t>::max() ||
        choices.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("pressure target arena is full");
    }
    const std::uint32_t offset = static_cast<std::uint32_t>(target_choice_arena.size());
    target_choice_arena.insert(target_choice_arena.end(), choices.begin(), choices.end());
    const std::uint32_t index = static_cast<std::uint32_t>(targets.size());
    targets.push_back(TargetNode{
        .candidate_index      = selected_candidate,
        .victim_choice_offset = offset,
        .victim_choice_count  = static_cast<std::uint32_t>(choices.size()),
        .stable_ordinal       = index,
        .root_maximal         = root_maximal,
    });
    index_target(index);
    return index;
}

void PressurePlanningSessionImpl::index_target(std::uint32_t target_index) {
    if (target_index >= targets.size() || target_hash_table.empty()) {
        throw std::logic_error("pressure target hash index is invalid");
    }
    const std::span<const std::uint16_t> choices = victim_choices(targets[target_index]);
    const std::size_t mask                       = target_hash_table.size() - 1U;
    std::size_t slot = static_cast<std::size_t>(planning_detail::target_hash(
                           targets[target_index].candidate_index, choices)) &
                       mask;
    for (std::size_t probe = 0; probe < target_hash_table.size(); ++probe) {
        std::uint32_t& indexed = target_hash_table[slot];
        if (indexed == std::numeric_limits<std::uint32_t>::max()) {
            indexed = target_index;
            return;
        }
        if (indexed < targets.size() &&
            targets[indexed].candidate_index == targets[target_index].candidate_index &&
            planning_detail::same_choices(victim_choices(targets[indexed]), choices)) {
            if (indexed != target_index) {
                throw std::logic_error("pressure target hash index is duplicated");
            }
            return;
        }
        slot = (slot + 1U) & mask;
    }
    throw std::length_error("pressure target hash table is full");
}

qwen3_5::PressureTargetHandle
PressurePlanningSessionImpl::identity_target(runtime::PlanningCandidateId candidate) const {
    qwen3_5::PressureTargetHandle handle;
    handle.session_    = this;
    handle.generation_ = generation;
    handle.index_      = candidate_index(candidate);
    return handle;
}

void PressurePlanningSessionImpl::populate_options(std::uint32_t selected_candidate) {
    if (selected_candidate >= candidate_options.size()) {
        throw std::out_of_range("pressure candidate index is invalid");
    }
    CandidateOptions& options = candidate_options[selected_candidate];
    if (options.populated) { return; }
    const CandidateState& candidate = *candidates[selected_candidate].state;
    const std::optional<Core::MaterializationSourceProtection> protection =
        program->materialization_source_protection(candidate);
    if (!protection) {
        throw std::logic_error("pressure planning candidate source protection is stale");
    }
    for (CandidateVictimOptions& victim : options.victims) {
        if (victim.owner_index >= owners.size()) {
            throw std::logic_error("pressure planning victim owner is invalid");
        }
        const Owner& owner                       = owners[victim.owner_index];
        std::vector<PressureDecision>& decisions = victim.decisions;
        using PlanningContractAccess             = qwen3_5::detail::RuntimeContractAccess;
        if (owner.shared) {
            PressureDecision eviction = program->inspect_shared_eviction_option(
                program->shared_prefix_states[PlanningContractAccess::index(*owner.shared_handle)]);
            if (!eviction.evicts_continuation || !eviction.shared_owner) {
                throw std::logic_error("shared pressure owner has no maximal outcome");
            }
            decisions.push_back(std::move(eviction));
        } else {
            PressureDecision eviction = program->inspect_eviction_option(
                program->continuation_states[PlanningContractAccess::index(*owner.private_handle)]);
            if (!eviction.evicts_continuation || eviction.shared_owner) {
                throw std::logic_error("private pressure owner has no maximal outcome");
            }
            decisions.push_back(std::move(eviction));
        }
        if (decisions.size() > std::numeric_limits<std::uint16_t>::max()) {
            throw std::overflow_error("pressure owner target count is not representable");
        }
        victim.eviction_choice = static_cast<std::uint16_t>(decisions.size());
    }
    options.populated = true;
}

std::vector<PressureDecision> PressurePlanningSessionImpl::pressure_successors(
    const CandidateVictimOptions& victim_options, const detail::PhysicalResources& residual,
    const Core::MaterializationSourceProtection& protection,
    const PressureDecision* current) const {
    if (victim_options.owner_index >= owners.size()) {
        throw std::out_of_range("pressure successor owner index is invalid");
    }

    using PlanningContractAccess = qwen3_5::detail::RuntimeContractAccess;
    std::vector<PressureDecision> successors;
    const Owner& victim_owner = owners[victim_options.owner_index];
    if (victim_owner.shared) {
        successors = program->inspect_shared_pressure_successors(
            program
                ->shared_prefix_states[PlanningContractAccess::index(*victim_owner.shared_handle)],
            residual, &protection, current);
    } else {
        successors = program->inspect_pressure_successors(
            program
                ->continuation_states[PlanningContractAccess::index(*victim_owner.private_handle)],
            residual, &protection, current);
    }
    if (victim_options.eviction_choice == 0 ||
        victim_options.eviction_choice > victim_options.decisions.size()) {
        throw std::logic_error("eligible pressure owner has no maximal outcome");
    }
    const PressureDecision& eviction =
        victim_options.decisions[victim_options.eviction_choice - 1U];
    if (std::find(successors.begin(), successors.end(), eviction) == successors.end()) {
        successors.push_back(eviction);
    }
    return successors;
}

qwen3_5::PressureTargetHandle
PressurePlanningSessionImpl::root_maximal_target(
    runtime::PlanningCandidateId root_candidate,
    std::span<const runtime::PlanningOwnerId> preferred_owner_ids,
    std::span<const std::uint32_t> preferred_owner_weights) {
    if (scratch_live) { throw std::logic_error("pressure expansion scratch is still live"); }
    const std::uint32_t selected_candidate = candidate_index(root_candidate);
    populate_options(selected_candidate);
    choice_scratch.assign(candidate_options[selected_candidate].victims.size(), 0);
    // The last-resort fallback must never destroy live sessions: only dead retained
    // weight (retention weight 0) is evicted, live victims are preserved so their
    // contexts survive the pressure episode and a following request reuses them.
    for (std::size_t index = 0; index < candidate_options[selected_candidate].victims.size();
         ++index) {
        const CandidateVictimOptions& victim =
            candidate_options[selected_candidate].victims[index];
        bool dead = true;
        for (std::size_t policy = 0; policy < preferred_owner_ids.size() &&
                                     policy < preferred_owner_weights.size();
             ++policy) {
            if (victim.owner_index < owners.size() &&
                owners[victim.owner_index].id == preferred_owner_ids[policy]) {
                dead = preferred_owner_weights[policy] == 0;
                break;
            }
        }
        if (dead && victim.eviction_choice != 0 &&
            victim.eviction_choice <= victim.decisions.size()) {
            choice_scratch[index] = victim.eviction_choice;
        }
    }
    const std::uint32_t target_index = intern_target(selected_candidate, choice_scratch, true);
    qwen3_5::PressureTargetHandle handle;
    handle.session_    = this;
    handle.generation_ = generation;
    handle.index_      = target_index;
    return handle;
}

qwen3_5::PressureTargetHandle
PressurePlanningSessionImpl::maximal_target(runtime::PlanningCandidateId id) {
    if (scratch_live) { throw std::logic_error("pressure expansion scratch is live"); }
    const auto selected = candidate_index(id);
    populate_options(selected);
    choice_scratch.clear();
    for (const auto& victim : candidate_options[selected].victims) {
        choice_scratch.push_back(victim.eviction_choice);
    }
    const auto index = intern_target(selected, choice_scratch);
    qwen3_5::PressureTargetHandle result;
    result.session_    = this;
    result.generation_ = generation;
    result.index_      = index;
    return result;
}

auto PressurePlanningSessionImpl::construction_slot(
    const qwen3_5::PressureConstructionCursor& cursor) -> ConstructionSlot& {
    if (cursor.session_ != this || cursor.slot_ >= construction_slots.size() || scratch_live ||
        resource_revision != program->resource_revision()) {
        throw std::logic_error("pressure construction cursor is stale");
    }
    auto& slot = construction_slots[cursor.slot_];
    if (!slot.leased || slot.generation != cursor.generation_) {
        throw std::logic_error("pressure construction lease is stale");
    }
    return slot;
}

void PressurePlanningSessionImpl::release_construction(const void* owner, std::uint32_t index,
                                                       std::uint32_t lease) noexcept {
    auto& session = *const_cast<PressurePlanningSessionImpl*>(
        static_cast<const PressurePlanningSessionImpl*>(owner));
    if (index < session.construction_slots.size()) {
        auto& slot = session.construction_slots[index];
        if (slot.generation == lease) {
            slot.leased = false;
            slot.options.clear();
        }
    }
}

detail::PhysicalResources
PressurePlanningSessionImpl::construction_residual(std::uint32_t index,
                                                   std::span<const std::uint16_t> choices) const {
    detail::PhysicalDelta pressure;
    const auto& options = candidate_options[index];
    for (std::size_t owner = 0; owner < choices.size(); ++owner) {
        if (choices[owner] == 0) { continue; }
        const auto& decision = options.victims[owner].decisions[choices[owner] - 1];
        pressure.added       = planning_resource_sum(pressure.added, decision.effect.added);
        pressure.removed     = planning_resource_sum(pressure.removed, decision.effect.removed);
    }
    auto result = program->guided_materialization_deficit(*candidates[index].state, pressure);
    result.host.kv_bytes =
        std::max(result.host.kv_bytes, candidates[index].state->blocked_host_allocation_bytes);
    return result;
}

qwen3_5::PressureConstructionCursor
PressurePlanningSessionImpl::begin_construction(qwen3_5::PressureTargetHandle target,
                                                bool restore) {
    if (!valid(target) || scratch_live) { throw std::logic_error("invalid construction parent"); }
    const auto& node = targets[target.index_];
    populate_options(node.candidate_index);
    for (std::uint32_t index = 0; index < construction_slots.size(); ++index) {
        auto& slot = construction_slots[index];
        if (slot.leased) { continue; }
        const auto choices = victim_choices(node);
        slot.choices.assign(choices.begin(), choices.end());
        slot.options.clear();
        slot.next_owner = slot.next_option = 0;
        slot.candidate_index               = node.candidate_index;
        slot.residual =
            node.assessed_residual.value_or(construction_residual(node.candidate_index, choices));
        slot.restore = restore;
        slot.leased  = true;
        if (++construction_generation == 0) { ++construction_generation; }
        slot.generation      = construction_generation;
        slot.scan_generation = 1;
        return qwen3_5::PressureConstructionCursor(this, index, slot.generation,
                                                   &release_construction);
    }
    throw std::length_error("all pressure construction cursors are leased");
}

runtime::PressureConstructionStep
PressurePlanningSessionImpl::next_construction_option(qwen3_5::PressureConstructionCursor& cursor) {
    auto& slot    = construction_slot(cursor);
    auto& options = candidate_options[slot.candidate_index];
    if (slot.next_option < slot.options.size()) {
        const auto index   = static_cast<std::uint32_t>(slot.next_option++);
        const auto& option = slot.options[index];
        return {.guidance = guidance_choices(slot.candidate_index, slot.choices, 0, option.victim,
                                             option.identity ? nullptr : &option.decision),
                .option   = {.cursor_generation = slot.generation,
                             .scan_generation   = slot.scan_generation,
                             .index             = index}};
    }
    if (slot.next_owner == options.victims.size()) { return {.exhausted = true}; }
    const auto owner_index          = slot.next_owner++;
    const auto& victim              = options.victims[owner_index];
    const auto choice               = slot.choices[owner_index];
    const PressureDecision* current = choice == 0 ? nullptr : &victim.decisions[choice - 1];
    const auto protection =
        program->materialization_source_protection(*candidates[slot.candidate_index].state);
    if (!protection) { throw std::logic_error("construction source protection is stale"); }
    const auto append = [&](PressureDecision decision, bool identity = false) {
        if (slot.options.size() == slot.options.capacity()) {
            throw std::length_error("construction option capacity exceeded");
        }
        slot.options.push_back(
            {.victim = owner_index, .decision = std::move(decision), .identity = identity});
    };
    if (slot.restore) {
        if (current != nullptr) {
            append({}, true);
            // Rebuild ordinary alternatives from the immutable candidate requirement, not from a
            // rewritten post-state. Complete assessment validates every less-destructive neighbor.
            for (auto& decision : pressure_successors(
                     victim, candidates[slot.candidate_index].state->identity_pressure_deficit,
                     *protection, nullptr)) {
                if (decision != *current && !decision.evicts_continuation) {
                    append(std::move(decision));
                }
            }
        }
    } else if (current == nullptr || !current->evicts_continuation) {
        for (auto& decision : pressure_successors(victim, slot.residual, *protection, current)) {
            if (current == nullptr || decision != *current) { append(std::move(decision)); }
        }
    }
    return {}; // one owner's successor generation is a separately metered operation
}

void PressurePlanningSessionImpl::choose_construction(qwen3_5::PressureConstructionCursor& cursor,
                                                      runtime::PressureConstructionOptionId id) {
    auto& slot = construction_slot(cursor);
    if (id.cursor_generation != slot.generation || id.scan_generation != slot.scan_generation ||
        id.index >= slot.options.size()) {
        throw std::logic_error("stale construction option");
    }
    auto& option         = slot.options[id.index];
    auto& decisions      = candidate_options[slot.candidate_index].victims[option.victim].decisions;
    std::uint16_t choice = 0;
    if (!option.identity) {
        const auto found = std::find(decisions.begin(), decisions.end(), option.decision);
        if (found != decisions.end()) {
            choice = static_cast<std::uint16_t>(1 + found - decisions.begin());
        } else {
            if (decisions.size() == std::numeric_limits<std::uint16_t>::max()) {
                throw std::length_error("construction owner decision capacity exceeded");
            }
            decisions.push_back(std::move(option.decision));
            choice = static_cast<std::uint16_t>(decisions.size());
        }
    }
    slot.choices[option.victim] = choice;
    slot.residual               = construction_residual(slot.candidate_index, slot.choices);
    slot.next_owner = slot.next_option = 0;
    slot.options.clear();
    if (++slot.scan_generation == 0) { ++slot.scan_generation; }
}

std::optional<qwen3_5::PressureTargetHandle> PressurePlanningSessionImpl::construction_target(
    const qwen3_5::PressureConstructionCursor& cursor) {
    auto& slot = construction_slot(cursor);
    if (!find_target(slot.candidate_index, slot.choices) &&
        targets.size() >= candidates.size() + 1U + planning_detail::kOptionalTargetCapacity) {
        return std::nullopt;
    }
    const auto index = intern_target(slot.candidate_index, slot.choices);
    qwen3_5::PressureTargetHandle result;
    result.session_    = this;
    result.generation_ = generation;
    result.index_      = index;
    return result;
}

std::optional<qwen3_5::PressureTargetHandle>
PressurePlanningSessionImpl::graceful_fallback_target(
    runtime::PlanningCandidateId admission,
    std::span<const runtime::PlanningOwnerId> preferred_owner_ids) {
    if (scratch_live) { throw std::logic_error("pressure expansion scratch is still live"); }
    const std::uint32_t selected_candidate = candidate_index(admission);
    populate_options(selected_candidate);
    CandidateOptions& options       = candidate_options[selected_candidate];
    const CandidateState& candidate = *candidates[selected_candidate].state;
    const std::optional<typename Core::MaterializationSourceProtection> protection =
        program->materialization_source_protection(candidate);
    if (!protection) { return std::nullopt; }

    std::vector<std::size_t> victim_order;
    victim_order.reserve(options.victims.size());
    const auto append_victim = [&](std::size_t victim_index) {
        if (std::find(victim_order.begin(), victim_order.end(), victim_index) ==
            victim_order.end()) {
            victim_order.push_back(victim_index);
        }
    };
    for (const runtime::PlanningOwnerId id : preferred_owner_ids) {
        const auto found =
            std::find_if(options.victims.begin(), options.victims.end(), [&](const auto& victim) {
                return victim.owner_index < owners.size() && owners[victim.owner_index].id == id;
            });
        if (found != options.victims.end()) {
            append_victim(static_cast<std::size_t>(found - options.victims.begin()));
        }
    }
    for (std::size_t index = 0; index < options.victims.size(); ++index) { append_victim(index); }

    const auto projected_residual = [&](std::span<const std::uint16_t> target_choices) {
        detail::PhysicalDelta pressure;
        for (std::size_t index = 0; index < options.victims.size(); ++index) {
            const std::uint16_t choice = target_choices[index];
            if (choice == 0) { continue; }
            if (choice > options.victims[index].decisions.size()) {
                throw std::logic_error("graceful pressure choice is invalid");
            }
            const PressureDecision& decision = options.victims[index].decisions[choice - 1U];
            pressure.added = planning_resource_sum(
                pressure.added, decision.effect.added);
            pressure.removed = planning_resource_sum(
                pressure.removed, decision.effect.removed);
        }
        return program->guided_materialization_deficit(candidate, pressure);
    };
    const auto feasible = [&](std::span<const std::uint16_t> target_choices) {
        return projected_residual(target_choices) == detail::PhysicalResources{};
    };

    choice_scratch.assign(options.victims.size(), 0);
    for (std::size_t index = 0; index < options.victims.size(); ++index) {
        choice_scratch[index] = options.victims[index].eviction_choice;
    }

    const auto choice_destructiveness = [&](std::size_t victim_index, std::uint16_t choice) {
        std::uint64_t score = 0;
        if (choice != 0) {
            const PressureDecision& decision = options.victims[victim_index].decisions[choice - 1U];
            score += decision.evicts_continuation ? 16ULL : 0ULL;
            score += decision.checkpoint_drops;
        }
        return score;
    };

    for (auto it = victim_order.rbegin(); it != victim_order.rend(); ++it) {
        const std::size_t victim_index = *it;
        if (choice_scratch[victim_index] == 0) { continue; }
        std::vector<PressureDecision>& decisions = options.victims[victim_index].decisions;
        {
            // Generate intermediate decisions (state/KV demotes) on demand so the
            // fallback can retain victims that only need a demote instead of a
            // whole-context drop (e.g. device state-slot pressure).
            const PressureDecision* current = nullptr;
            if (choice_scratch[victim_index] <= decisions.size()) {
                current = &decisions[choice_scratch[victim_index] - 1U];
            }
            const detail::PhysicalResources residual = projected_residual(choice_scratch);
            for (PressureDecision& successor :
                 pressure_successors(options.victims[victim_index], residual, *protection,
                                     current)) {
                if (std::find(decisions.begin(), decisions.end(), successor) != decisions.end()) {
                    continue;
                }
                if (decisions.size() >= std::numeric_limits<std::uint16_t>::max()) { break; }
                decisions.push_back(std::move(successor));
            }
        }
        std::uint16_t best_choice = choice_scratch[victim_index];
        std::uint64_t best_score  = choice_destructiveness(victim_index, best_choice);
        for (std::uint16_t choice = 0; choice <= decisions.size(); ++choice) {
            if (choice == best_choice) { continue; }
            choice_scratch[victim_index] = choice;
            if (!feasible(choice_scratch)) { continue; }
            const std::uint64_t score = choice_destructiveness(victim_index, choice);
            if (score < best_score) {
                best_score  = score;
                best_choice = choice;
            }
        }
        choice_scratch[victim_index] = best_choice;
    }

    TargetNode* existing = find_target(selected_candidate, choice_scratch);
    const std::size_t maximum =
        candidates.size() + 1U + planning_detail::kOptionalTargetCapacity;
    if (existing == nullptr && targets.size() >= maximum) { return std::nullopt; }
    const std::uint32_t target_index =
        existing != nullptr ? static_cast<std::uint32_t>(existing - targets.data())
                            : intern_target(selected_candidate, choice_scratch);
    qwen3_5::PressureTargetHandle handle;
    handle.session_    = this;
    handle.generation_ = generation;
    handle.index_      = target_index;
    return handle;
}

std::optional<qwen3_5::PressureTargetHandle>
PressurePlanningSessionImpl::deterministic_target(
    runtime::PlanningCandidateId admission,
    std::span<const runtime::PlanningOwnerId> preferred_owner_ids,
    std::span<const std::uint32_t> preferred_owner_weights,
    std::span<const std::uint64_t> preferred_owner_epochs) {
    if (preferred_owner_ids.size() != preferred_owner_weights.size() ||
        preferred_owner_ids.size() != preferred_owner_epochs.size()) {
        throw std::logic_error("pressure preferred owner arrays are misaligned");
    }
    if (scratch_live) { throw std::logic_error("pressure expansion scratch is still live"); }
    const std::uint32_t selected_candidate = candidate_index(admission);
    populate_options(selected_candidate);
    CandidateOptions& options       = candidate_options[selected_candidate];
    const CandidateState& candidate = *candidates[selected_candidate].state;
    const std::optional<typename Core::MaterializationSourceProtection> protection =
        program->materialization_source_protection(candidate);
    if (!protection) { return std::nullopt; }

    // Victims ranked least-valuable first: preferred_owner_ids is already sorted by
    // ascending value (selected_hit_count, shared credit, retention weight, recency);
    // unranked victims trail in owner order. The ranking drives which victims are
    // dropped when both tiers are full; demotion preserves every victim regardless.
    std::vector<std::size_t> victim_order;
    victim_order.reserve(options.victims.size());
    const auto append_victim = [&](std::size_t victim_index) {
        if (std::find(victim_order.begin(), victim_order.end(), victim_index) ==
            victim_order.end()) {
            victim_order.push_back(victim_index);
        }
    };
    for (const runtime::PlanningOwnerId id : preferred_owner_ids) {
        const auto found =
            std::find_if(options.victims.begin(), options.victims.end(), [&](const auto& victim) {
                return victim.owner_index < owners.size() && owners[victim.owner_index].id == id;
            });
        if (found != options.victims.end()) {
            append_victim(static_cast<std::size_t>(found - options.victims.begin()));
        }
    }
    for (std::size_t index = 0; index < options.victims.size(); ++index) { append_victim(index); }

    std::fprintf(stderr, "[PP] plan cand=%u demand_pages=%u victims=%zu owners=",
                 selected_candidate, candidate.demand.active_entitlement.device.main_kv_pages,
                 options.victims.size());
    for (std::size_t index = 0; index < options.victims.size(); ++index) {
        const Owner& owner = owners[options.victims[index].owner_index];
        std::fprintf(stderr, "%s%u%s", index == 0 ? "" : ",", owner.id.value,
                     owner.shared ? "s" : "p");
    }
    std::fprintf(stderr, "\n");

    const auto projected_residual = [&](std::span<const std::uint16_t> target_choices,
                                        std::optional<std::size_t> override_owner,
                                        const PressureDecision* override_decision) {
        detail::PhysicalDelta pressure;
        std::vector<std::uint32_t> demote_main;
        std::vector<std::uint32_t> demote_back;
        for (std::size_t index = 0; index < options.victims.size(); ++index) {
            const PressureDecision* decision = nullptr;
            if (override_owner && *override_owner == index) {
                decision = override_decision;
            } else {
                const std::uint16_t choice = target_choices[index];
                if (choice != 0) {
                    if (choice > options.victims[index].decisions.size()) {
                        throw std::logic_error("deterministic pressure choice is invalid");
                    }
                    decision = &options.victims[index].decisions[choice - 1U];
                }
            }
            if (decision == nullptr) { continue; }
            pressure.added = planning_resource_sum(
                pressure.added, decision->effect.added);
            pressure.removed = planning_resource_sum(
                pressure.removed, decision->effect.removed);
            for (const PressureKVDecision& kv : decision->main_kv_changes) {
                if (kv.kind == PressureKVDecisionKind::DemoteToHost && kv.page_count != 0) {
                    demote_main.push_back(kv.page_count);
                }
            }
            for (const PressureKVDecision& kv : decision->backend_kv_changes) {
                if (kv.kind == PressureKVDecisionKind::DemoteToHost && kv.page_count != 0) {
                    demote_back.push_back(kv.page_count);
                }
            }
        }
        detail::PhysicalResources residual =
            program->guided_materialization_deficit(candidate, pressure);
        // The plan's demotes add Host KV that compose_pressure_candidate must actually
        // allocate in the Host extent arena, whose free space can be fragmented (the
        // byte-total model above is optimistic). Ask the allocator: if the demote
        // requests fit the current free extents, the byte model stands; otherwise the
        // unpaid Host addition (demoted minus freed-by-DropHostDuplicate) becomes Host
        // pressure so Phase 2 pays it by dropping Host duplicates of low-value shared
        // prefixes — demote without eviction, reprocessing nothing.
        const std::uint64_t added_host_kv = pressure.added.host.kv_bytes;
        const std::uint64_t freed_host_kv = pressure.removed.host.kv_bytes;
        // The admission's blocked bytes are all-or-nothing (the arena cannot serve the
        // whole demote set as-is), but the tier still has free space. Charge only the
        // true shortfall: requested minus the currently free Host bytes minus anything
        // this plan already frees. Otherwise Phase 2 would over-evict (targeting the
        // full demote instead of the few stale contexts that actually cover the gap).
        const detail::PhysicalResources host_occupancy = program->physical_occupancy();
        const std::uint64_t host_capacity_bytes =
            program->admission_capacity().host.kv_bytes;
        const std::uint64_t host_free_bytes =
            host_capacity_bytes > host_occupancy.host.kv_bytes
                ? host_capacity_bytes - host_occupancy.host.kv_bytes
                : 0;
        const std::uint64_t host_shortfall =
            added_host_kv > freed_host_kv + host_free_bytes
                ? added_host_kv - freed_host_kv - host_free_bytes
                : 0;
        // The byte model above over-credits Host frees the hot tier never lets go
        // of (host copies are kept, so compose's real allocator can reject a demote
        // set the model considers free — run 44: 2.79 GiB requested vs 2.48 GiB free,
        // residual host=0). Ask the allocator directly: if the plan's demotes don't
        // fit the current free extents, surface the true Host demand as pressure so
        // Phase 3 converts demotes to evictions (zero Host demand) instead of
        // compose failing the whole plan and falling back to a root re-prefill.
        bool host_fits            = true;
        bool allocator_fits       = false;
        bool deterministic_cover  = false;
        if (added_host_kv != 0) {
            // The conservative allocator answer (no releases) is safe but blind to the
            // Host this plan itself frees by eviction or DropHostDuplicate — and both
            // are deterministic at compose (a destroyed victim's Host extents, or the
            // release of a device+Host duplicate). At high Host occupancy that blind
            // spot wrongly blocks a relief plan and the planner falls back to a root
            // re-prefill. Accept when the real allocator fits OR the plan's own
            // deterministic Host frees cover the demote demand.
            allocator_fits = program->host_kv_requests_fit(demote_main, demote_back);
            deterministic_cover = added_host_kv <= host_free_bytes + freed_host_kv;
            host_fits = allocator_fits || deterministic_cover;
        }
        // Host pressure when the plan does not fit is the deterministic shortfall: the
        // demote demand minus what the plan's own deterministic Host frees (evictions
        // and DropHostDuplicate) already free minus the tier's free space. Charging the
        // full demand here (all-or-nothing) hid partial relief — chained small frees
        // could never accumulate in the residual key, so the planner jumped straight to
        // a single big victim (a live main) instead of chaining dead-weight relief.
        // DropHostDuplicate is credited here too: the probe data shows device+Host
        // duplicates are real (hundreds of pages per victim), and dropping them frees
        // Host without touching a live context.
        const std::uint64_t host_det_shortfall =
            added_host_kv > freed_host_kv + host_free_bytes
                ? added_host_kv - freed_host_kv - host_free_bytes
                : 0;
        if (host_fits) {
            residual.host.kv_bytes = std::max(residual.host.kv_bytes, host_shortfall);
        } else {
            residual.host.kv_bytes = std::max(residual.host.kv_bytes, host_det_shortfall);
        }
        std::fprintf(stderr,
                     "[PP] hostfit add=%zu free=%zu short=%zu det=%zu alloc=%d evcover=%d fits=%d\n",
                     added_host_kv, host_free_bytes, host_shortfall, host_det_shortfall,
                     allocator_fits ? 1 : 0, deterministic_cover ? 1 : 0,
                     host_fits ? 1 : 0);
        return residual;
    };
    const detail::PhysicalResources capacity = program->admission_capacity();
    constexpr std::uint64_t kResidualOne     = 1ULL << 20U;
    const auto normalized                    = [](std::uint64_t value, std::uint64_t limit) {
        if (value == 0) { return std::uint64_t{0}; }
        if (limit == 0 || value >= limit) { return kResidualOne; }
        if (value > std::numeric_limits<std::uint64_t>::max() / kResidualOne) {
            return kResidualOne;
        }
        const std::uint64_t scaled = value * kResidualOne;
        return std::max<std::uint64_t>(1, scaled / limit + (scaled % limit != 0 ? 1U : 0U));
    };
    const auto residual_key = [&](const detail::PhysicalResources& residual) {
        std::uint32_t constraints = 0;
        std::uint64_t total       = 0;
        const auto append         = [&](std::uint64_t value, std::uint64_t limit) {
            if (value == 0) { return; }
            ++constraints;
            planning_saturating_add(total, normalized(value, limit));
        };
        append(residual.device.active_lanes, capacity.device.active_lanes);
        append(residual.device.state_slots, capacity.device.state_slots);
        append(residual.device.main_kv_pages, capacity.device.main_kv_pages);
        append(residual.device.backend_kv_pages, capacity.device.backend_kv_pages);
        append(residual.host.state_slots, capacity.host.state_slots);
        append(residual.host.kv_bytes, capacity.host.kv_bytes);
        return std::tuple{constraints, total};
    };
    const auto feasible = [&](const detail::PhysicalResources& residual) {
        return residual == detail::PhysicalResources{};
    };
    // Device and Host over-commitment keys drive the two-stage waterfall: demotes may
    // defer Host over-commitment (paid off by state drops) instead of being blocked by
    // a momentarily full Host tier.
    const auto device_key = [&](const detail::PhysicalResources& residual) {
        std::uint64_t total = 0;
        const auto append   = [&](std::uint64_t value, std::uint64_t limit) {
            if (value == 0) { return; }
            planning_saturating_add(total, normalized(value, limit));
        };
        append(residual.device.active_lanes, capacity.device.active_lanes);
        append(residual.device.state_slots, capacity.device.state_slots);
        append(residual.device.main_kv_pages, capacity.device.main_kv_pages);
        append(residual.device.backend_kv_pages, capacity.device.backend_kv_pages);
        return total;
    };

    struct Selection {
        std::size_t victim_index = 0;
        PressureDecision decision;
        detail::PhysicalResources residual;
    };
    const auto intern_selection = [&](const Selection& selection) {
        std::vector<PressureDecision>& decisions =
            options.victims[selection.victim_index].decisions;
        const auto existing =
            std::find(decisions.begin(), decisions.end(), selection.decision);
        if (existing != decisions.end()) {
            return static_cast<std::uint16_t>(1U + (existing - decisions.begin()));
        }
        if (decisions.size() >= std::numeric_limits<std::uint16_t>::max()) {
            throw std::length_error("pressure owner target count is not representable");
        }
        decisions.push_back(selection.decision);
        return static_cast<std::uint16_t>(decisions.size());
    };
    const auto refresh_options = [&](CandidateVictimOptions& victim,
                                     const detail::PhysicalResources& residual,
                                     const PressureDecision* current) {
        std::vector<PressureDecision> successors =
            pressure_successors(victim, residual, *protection, current);
        for (PressureDecision& successor : successors) {
            if (std::find(victim.decisions.begin(), victim.decisions.end(), successor) ==
                victim.decisions.end()) {
                if (victim.decisions.size() >= std::numeric_limits<std::uint16_t>::max()) {
                    throw std::length_error("pressure owner target count is not representable");
                }
                victim.decisions.push_back(std::move(successor));
            }
        }
    };

    choice_scratch.assign(options.victims.size(), 0);
    const std::size_t maximum_steps =
        32U * std::max<std::size_t>(1, options.victims.size()) + 32U;
    using PlanningContractAccess =
        RuntimeContractAccess;
    const auto owner_kv = [&](std::size_t victim_index)
        -> std::pair<std::optional<KVAddressSpaceHandle>,
                     std::optional<KVAddressSpaceHandle>> {
        const Owner& owner = owners[options.victims[victim_index].owner_index];
        if (owner.shared) {
            const auto& s =
                program->shared_prefix_states[PlanningContractAccess::index(*owner.shared_handle)];
            if (!s.kv) { return {}; }
            return {s.kv->text, s.kv->backend};
        }
        const auto& c =
            program->continuation_states[PlanningContractAccess::index(*owner.private_handle)];
        if (!c.kv) { return {}; }
        return {c.kv->text, c.kv->backend};
    };
    // Page-disjointness guard: demote decisions must never target the same
    // physical KV page twice within one plan. Nested same-session shared
    // prefixes (and the shared system prefix) overlap, and the exact
    // materialization check rejects duplicated page targets — which would
    // otherwise collapse the whole deterministic plan and fall back to root.
    std::vector<std::vector<std::uint32_t>> victim_pages(options.victims.size());
    const auto collect_pages = [&](std::size_t victim_index,
                                   const PressureDecision& decision,
                                   std::vector<std::uint32_t>& text_targets,
                                   std::vector<std::uint32_t>& back_targets) -> bool {
        const auto [text_address, back_address] = owner_kv(victim_index);
        const auto collect = [&](const auto& addresses, const auto& pages,
                                 std::optional<KVAddressSpaceHandle> address,
                                 const std::vector<PressureKVDecision>& changes,
                                 std::vector<std::uint32_t>& out) -> bool {
            if (changes.empty()) { return true; }
            if (!address || !addresses || !pages) { return false; }
            const std::uint32_t mapped = addresses->mapped_pages(*address);
            for (const PressureKVDecision& action : changes) {
                if (action.kind == PressureKVDecisionKind::None) { continue; }
                if (action.begin_page > mapped ||
                    action.page_count > mapped - action.begin_page) {
                    return false;
                }
                for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
                    out.push_back(pages->descriptor_index(
                        addresses->logical_page(*address, action.begin_page + offset)));
                }
            }
            return true;
        };
        return collect(program->text_kv_addresses, program->text_kv_pages, text_address,
                       decision.main_kv_changes, text_targets) &&
               collect(program->backend_kv_addresses, program->backend_kv_pages, back_address,
                       decision.backend_kv_changes, back_targets);
    };
    const auto overlaps_committed = [&](std::size_t victim_index,
                                        const PressureDecision& decision) -> bool {
        std::vector<std::uint32_t> text_targets;
        std::vector<std::uint32_t> back_targets;
        if (!collect_pages(victim_index, decision, text_targets, back_targets)) { return true; }
        const auto overlaps = [&](const std::vector<std::uint32_t>& candidate,
                                  const std::vector<std::uint32_t>& committed) {
            for (const std::uint32_t page : candidate) {
                if (std::find(committed.begin(), committed.end(), page) != committed.end()) {
                    return true;
                }
            }
            return false;
        };
        for (std::size_t other = 0; other < options.victims.size(); ++other) {
            if (other == victim_index) { continue; }
            const bool text_overlap  = overlaps(text_targets, victim_pages[other]);
            const bool back_overlap  = overlaps(back_targets, victim_pages[other]);
            if (text_overlap || back_overlap) {
                return true;
            }
        }
        return false;
    };
    const auto commit_pages = [&](std::size_t victim_index, const PressureDecision& decision) {
        std::vector<std::uint32_t> text_targets;
        std::vector<std::uint32_t> back_targets;
        if (!collect_pages(victim_index, decision, text_targets, back_targets)) {
            throw std::logic_error("deterministic page guard cannot resolve target pages");
        }
        std::vector<std::uint32_t>& own = victim_pages[victim_index];
        own.clear();
        for (const std::uint32_t page : text_targets) {
            if (std::find(own.begin(), own.end(), page) == own.end()) { own.push_back(page); }
        }
        for (const std::uint32_t page : back_targets) {
            if (std::find(own.begin(), own.end(), page) == own.end()) { own.push_back(page); }
        }
    };


    // Trim a demote/device-drop decision to its exclusive pages: drop pages already
    // committed by other victims (shared prefixes / nested contexts). A whole-tree
    // demote (main + shared prefix) would otherwise be rejected by the page guard
    // and the victim would fall through to eviction — the exact bug that made the
    // planner mass-evict mains at the ~130K working-set point.
    const auto trim_to_exclusive = [&](std::size_t victim_index,
                                       const PressureDecision& decision)
        -> std::optional<PressureDecision> {
        std::vector<std::uint32_t> committed;
        for (std::size_t other = 0; other < options.victims.size(); ++other) {
            if (other == victim_index) { continue; }
            committed.insert(committed.end(), victim_pages[other].begin(),
                             victim_pages[other].end());
        }
        std::sort(committed.begin(), committed.end());
        committed.erase(std::unique(committed.begin(), committed.end()), committed.end());

        const auto [text_addr, back_addr] = owner_kv(victim_index);

        PressureDecision out = decision;
        out.main_kv_changes.clear();
        out.backend_kv_changes.clear();
        out.transfer_requirements.clear();
        out.effect = detail::PhysicalDelta{};
        // State/checkpoint contributions are untouched by the trim.
        out.effect.removed.device.state_slots = decision.effect.removed.device.state_slots;
        out.effect.added.host.state_slots     = decision.effect.added.host.state_slots;
        out.effect.removed.host.state_slots   = decision.effect.removed.host.state_slots;
        out.effect.added.device.state_slots   = decision.effect.added.device.state_slots;
        out.checkpoint_drop_effect            = decision.checkpoint_drop_effect;

        const auto trim_store = [&](const auto& addresses, const auto& pages,
                                    std::optional<KVAddressSpaceHandle> address,
                                    const std::vector<PressureKVDecision>& changes,
                                    std::vector<PressureKVDecision>& out_changes,
                                    runtime::ContextResourceClass resource) {
            if (!address || !addresses || !pages) { return; }
            const HostKVPageLayout layout =
                plan_host_kv_page_layout(pages->physical_pool().geometry());
            const std::uint32_t mapped = addresses->mapped_pages(*address);
            const bool backend =
                resource == runtime::ContextResourceClass::BackendKV;
            const auto emit = [&](PressureKVDecision action) {
                const std::uint64_t bytes =
                    layout.page_stride * static_cast<std::uint64_t>(action.page_count);
                if (action.kind == PressureKVDecisionKind::DemoteToHost) {
                    if (backend) {
                        out.effect.removed.device.backend_kv_pages += action.page_count;
                    } else {
                        out.effect.removed.device.main_kv_pages += action.page_count;
                    }
                    out.effect.added.host.kv_bytes += bytes;
                    std::vector<DeviceKVPageHandle> physical;
                    physical.reserve(action.page_count);
                    for (std::uint32_t off = 0; off < action.page_count; ++off) {
                        physical.push_back(pages->physical(
                            addresses->logical_page(*address, action.begin_page + off)));
                    }
                    const std::uint32_t runs =
                        pages->physical_pool().contiguous_run_count(physical);
                    const TransferWork work =
                        plan_host_kv_transfer_work(layout, action.page_count, runs);
                    out.transfer_requirements.push_back(runtime::ContextTransferRequirement{
                        .resource   = resource,
                        .direction  = runtime::ContextTransferDirection::DeviceToHost,
                        .units      = work.payload_bytes,
                        .page_count = action.page_count,
                        .work       = work,
                    });
                } else if (action.kind ==
                           PressureKVDecisionKind::DropDeviceDuplicate) {
                    if (backend) {
                        out.effect.removed.device.backend_kv_pages += action.page_count;
                    } else {
                        out.effect.removed.device.main_kv_pages += action.page_count;
                    }
                } else if (action.kind == PressureKVDecisionKind::DropHostDuplicate) {
                    out.effect.removed.host.kv_bytes += bytes;
                }
                out_changes.push_back(action);
            };
            for (const PressureKVDecision& action : changes) {
                if (action.kind == PressureKVDecisionKind::None || action.page_count == 0) {
                    continue;
                }
                std::uint32_t run_begin = action.begin_page;
                const std::uint32_t end = action.begin_page + action.page_count;
                for (std::uint32_t p = action.begin_page; p < end; ++p) {
                    const LogicalKVPageHandle logical =
                        addresses->logical_page(*address, p);
                    const bool taken = std::binary_search(
                        committed.begin(), committed.end(), pages->descriptor_index(logical));
                    if (taken) {
                        if (p > run_begin) {
                            emit({.begin_page = run_begin, .page_count = p - run_begin,
                                  .kind = action.kind});
                        }
                        run_begin = p + 1;
                    }
                }
                if (end > run_begin) {
                    emit({.begin_page = run_begin, .page_count = end - run_begin,
                          .kind = action.kind});
                }
            }
        };
        trim_store(program->text_kv_addresses, program->text_kv_pages, text_addr,
                   decision.main_kv_changes, out.main_kv_changes,
                   runtime::ContextResourceClass::MainKV);
        trim_store(program->backend_kv_addresses, program->backend_kv_pages, back_addr,
                   decision.backend_kv_changes, out.backend_kv_changes,
                   runtime::ContextResourceClass::BackendKV);
        if (out.main_kv_changes.empty() && out.backend_kv_changes.empty() &&
            out.state_changes.empty() && out.dropped_checkpoints.empty()) {
            return std::nullopt;
        }
        return out;
    };

    const auto pp_selection = [&](const char* tag, const Selection& sel) {
        const PressureDecision& d = sel.decision;
        const Owner& o = owners[options.victims[sel.victim_index].owner_index];
        const auto victim_kv = owner_kv(sel.victim_index);
        std::uint32_t vmain = 0;
        if (victim_kv.first && program->text_kv_addresses) {
            vmain = program->text_kv_addresses->mapped_pages(*victim_kv.first);
        }
        std::fprintf(stderr,
                     "[PP] %s victim=%u%s evict=%d main_pages=%u dev_main_rem=%u dev_back_rem=%u "
                     "state_rem=%u host_add=%zu host_rem=%zu drops=%u\n",
                     tag, o.id.value, o.shared ? "s" : "p", d.evicts_continuation ? 1 : 0, vmain,
                     d.effect.removed.device.main_kv_pages,
                     d.effect.removed.device.backend_kv_pages,
                     d.effect.removed.device.state_slots, d.effect.added.host.kv_bytes,
                     d.effect.removed.host.kv_bytes, d.checkpoint_drops);
    };

    // Victim size in main+backend pages, used only as the final tiebreak of the
    // value-ranked eviction order below.
    const auto victim_size = [&](std::size_t victim_index) -> std::uint64_t {
        const auto kv = owner_kv(victim_index);
        std::uint64_t pages = 0;
        if (kv.first && program->text_kv_addresses) {
            pages += program->text_kv_addresses->mapped_pages(*kv.first);
        }
        if (kv.second && program->backend_kv_addresses) {
            pages += program->backend_kv_addresses->mapped_pages(*kv.second);
        }
        return pages;
    };
    std::vector<std::size_t> eviction_order = victim_order;
    // Value-ranked eviction order: retention weight is the primary key (shared/absent
    // weight sheds first, recently-active live sessions last), then recency (least
    // recently hit sheds first; a never-hit epoch-0 victim is freshly admitted and
    // protected, matching the planner's ranking), then size as a final tiebreak so the
    // least destructive victim of a class is evicted first. Recency is what actually
    // separates live sessions from dead retained weight when both carry the same
    // retention weight (e.g. RecentPrivate): an actively-stepping session's epoch
    // advances every turn, a long-idle retained context's is frozen, so the dead weight
    // sheds before a live main regardless of size. A pure size key does the opposite —
    // the live mains are the largest RecentPrivate victims and get shed before the small
    // dead contexts, which is the ~130K working-set mass-eviction defect.
    std::vector<std::uint32_t> victim_weights(options.victims.size(), 0);
    std::vector<std::uint64_t> victim_epochs(options.victims.size(), 0);
    for (std::size_t index = 0;
         index < preferred_owner_ids.size() && index < preferred_owner_weights.size() &&
         index < preferred_owner_epochs.size();
         ++index) {
        const auto found = std::find_if(
            options.victims.begin(), options.victims.end(), [&](const auto& victim) {
                return victim.owner_index < owners.size() &&
                       owners[victim.owner_index].id == preferred_owner_ids[index];
            });
        if (found != options.victims.end()) {
            victim_weights[static_cast<std::size_t>(found - options.victims.begin())] =
                preferred_owner_weights[index];
            victim_epochs[static_cast<std::size_t>(found - options.victims.begin())] =
                preferred_owner_epochs[index];
        }
    }
    std::stable_sort(eviction_order.begin(), eviction_order.end(),
                     [&](std::size_t left, std::size_t right) {
                         const std::uint32_t left_weight  = victim_weights[left];
                         const std::uint32_t right_weight = victim_weights[right];
                         if (left_weight != right_weight) { return left_weight < right_weight; }
                         const std::uint64_t left_epoch =
                             victim_epochs[left] == 0 ? ~std::uint64_t{0} : victim_epochs[left];
                         const std::uint64_t right_epoch =
                             victim_epochs[right] == 0 ? ~std::uint64_t{0}
                                                       : victim_epochs[right];
                         if (left_epoch != right_epoch) { return left_epoch < right_epoch; }
                         return victim_size(left) < victim_size(right);
                     });
    // Live-vs-dead split for the Host-placement fix-up below. The retention-weight
    // escalation (RecentPrivate -> LiveSession) engages once an owner accumulates
    // enough recorded hits; recency is the primary split here because it also
    // distinguishes actively-stepping sessions from long-idle retained weight when
    // both carry the same retention weight (e.g. RecentPrivate): an actively-stepping
    // session's epoch advances every turn, a long-idle retained context's is frozen,
    // so a victim whose last hit sits within the activity window of the freshest
    // owner is live (demote, never evict) and everything older is dead retained
    // weight (shed first). Mirrors the resource-manager activity window that the
    // weight escalation uses.
    constexpr std::uint64_t kPlannerActivityWindow = 32;
    std::uint64_t freshest_epoch                   = 0;
    for (std::size_t index = 0; index < options.victims.size(); ++index) {
        freshest_epoch = std::max(freshest_epoch, victim_epochs[index]);
    }
    {
        std::fprintf(stderr, "[PP] epochs freshest=%llu gap32",
                     static_cast<unsigned long long>(freshest_epoch));
        for (std::size_t index = 0; index < options.victims.size(); ++index) {
            const Owner& owner = owners[options.victims[index].owner_index];
            const std::uint64_t gap =
                victim_epochs[index] == 0
                    ? 0
                    : (freshest_epoch >= victim_epochs[index]
                           ? freshest_epoch - victim_epochs[index]
                           : 0);
            std::fprintf(stderr, " %u%s(e%llu,g%llu,w%u)", owner.id.value, owner.shared ? "s" : "p",
                         static_cast<unsigned long long>(victim_epochs[index]),
                         static_cast<unsigned long long>(gap), victim_weights[index]);
        }
        std::fprintf(stderr, "\n");
        if (program->text_kv_pages != nullptr) {
            const auto tr = program->text_kv_pages->replica_residency_counts();
            const auto br = program->backend_kv_pages != nullptr
                                ? program->backend_kv_pages->replica_residency_counts()
                                : qwen3_5::detail::LogicalKVPageStore::ReplicaResidencyCounts{};
            std::fprintf(stderr, "[PP] resid t(d=%u,h=%u,b=%u) k(d=%u,h=%u,b=%u) host=%zu\n",
                         tr.device_only, tr.host_only, tr.both, br.device_only, br.host_only,
                         br.both, static_cast<std::size_t>(program->physical_occupancy().host.kv_bytes));
        }
    }
    const auto victim_live = [&](std::size_t victim_index) {
        const std::uint64_t epoch = victim_epochs[victim_index];
        if (epoch == 0) {
            // A never-hit victim is freshly admitted and protected — except shared
            // stable prefixes. A shared owner's hit epoch is not recorded, so it
            // stays 0 forever and would otherwise be locked in as "freshly admitted"
            // while holding device+Host that live sessions need. Shared (weight-0)
            // victims are therefore shed-able regardless of epoch.
            return victim_weights[victim_index] != 0;
        }
        return freshest_epoch >= epoch && freshest_epoch - epoch <= kPlannerActivityWindow;
    };

    // Phase 1 — device waterfall: while the device is over-committed, demote the
    // lowest-value victims whose retained (non-evicting, no-new-checkpoint-drop)
    // decision strictly reduces device over-commitment. Host impact is deferred to
    // phase 2, so several demotes can chain even when the Host tier is momentarily full.
    for (std::size_t step = 0; step < maximum_steps; ++step) {
        const detail::PhysicalResources residual =
            projected_residual(choice_scratch, std::nullopt, nullptr);
        if (device_key(residual) == 0) { break; }
        std::optional<Selection> selected;
        for (const std::size_t victim_index : victim_order) {
            CandidateVictimOptions& victim = options.victims[victim_index];
            const std::uint16_t current_choice = choice_scratch[victim_index];
            const PressureDecision* current =
                current_choice == 0 ? nullptr : &victim.decisions[current_choice - 1U];
            if (current != nullptr && current->evicts_continuation) { continue; }
            refresh_options(victim, residual, current);
            for (std::uint16_t choice = 1; choice <= victim.decisions.size(); ++choice) {
                const PressureDecision& candidate = victim.decisions[choice - 1U];
                if (candidate.evicts_continuation) { continue; }
                const std::uint32_t prior_drops =
                    current == nullptr ? 0 : current->checkpoint_drops;
                if (candidate.checkpoint_drops > prior_drops) { continue; }
                const PressureDecision* effective = &candidate;
                PressureDecision trimmed;
                if (overlaps_committed(victim_index, candidate)) {
                    std::optional<PressureDecision> t =
                        trim_to_exclusive(victim_index, candidate);
                    if (!t || (t->main_kv_changes.empty() && t->backend_kv_changes.empty() &&
                               t->state_changes.empty())) {
                        continue;
                    }
                    trimmed = std::move(*t);
                    effective = &trimmed;
                }
                const detail::PhysicalResources child =
                    projected_residual(choice_scratch, victim_index, effective);
                if (device_key(child) >= device_key(residual)) { continue; }
                selected = Selection{
                    .victim_index = victim_index,
                    .decision     = *effective,
                    .residual     = child,
                };
                break;
            }
            if (selected) { break; }
        }
        if (!selected) { break; }
        pp_selection("P1", *selected);
        choice_scratch[selected->victim_index] = intern_selection(*selected);
        commit_pages(selected->victim_index, selected->decision);
    }

    // Phase 2 — Host payoff: while Host (or the whole plan) is still over-committed,
    // drop the state checkpoints of the lowest-value victims whose retained decision
    // adds checkpoint drops and strictly reduces the total over-commitment. This pays
    // the Host debt deferred by phase 1 (and any Host pressure the admission itself
    // brings) without evicting any context — KV stays cached, prefixes stay reusable.
    // Victims are considered in eviction order (smallest first) so a host-relief
    // eviction takes the least destructive victim available, not the lowest ID.
    for (std::size_t step = 0; step < maximum_steps; ++step) {
        const detail::PhysicalResources residual =
            projected_residual(choice_scratch, std::nullopt, nullptr);
        if (feasible(residual)) { break; }
        std::optional<Selection> selected;
        for (const std::size_t victim_index : eviction_order) {
            CandidateVictimOptions& victim = options.victims[victim_index];
            const std::uint16_t current_choice = choice_scratch[victim_index];
            const PressureDecision* current =
                current_choice == 0 ? nullptr : &victim.decisions[current_choice - 1U];
            if (current != nullptr && current->evicts_continuation) { continue; }
            refresh_options(victim, residual, current);
            for (std::uint16_t choice = 1; choice <= victim.decisions.size(); ++choice) {
                const PressureDecision& candidate = victim.decisions[choice - 1U];
                const std::uint32_t prior_drops =
                    current == nullptr ? 0 : current->checkpoint_drops;
                if (candidate.evicts_continuation) {
                    // Host-relief eviction: relieve a full Host tier (and the device
                    // over-commitment) by evicting the lowest-value victim whose eviction
                    // frees Host bytes. Private victims only — the shared system prefix
                    // and live mains' shared base must survive (their pages are not
                    // exclusive to this owner anyway). A demoted victim may be evicted too,
                    // and is in fact the preferred target: the dead subs Phase 1 demoted
                    // have ~0 exclusive Host (their pages overlap the shared prefix), but
                    // replacing their demote with an eviction drops the whole demote's Host
                    // demand — closing the squeeze without touching a live main. Smallest
                    // victims are tried first.
                    const bool removes_demote_host =
                        current != nullptr && current->effect.added.host.kv_bytes != 0;
                    if (candidate.effect.removed.host.kv_bytes == 0 && !removes_demote_host) {
                        continue;
                    }
                    if (owners[options.victims[victim_index].owner_index].shared) { continue; }
                    // Live victims are never host-relief evicted here: their overflow is
                    // demoted (or shed) by the later phases, which can spend the Host that
                    // these dead-weight evictions free. Evicting an active session here
                    // destroys a prefix a demote could have preserved — and it ranks before
                    // P4's fix-up, whose evict branch already applies this same liveness
                    // split.
                    if (victim_live(victim_index)) { continue; }
                } else {
                    const bool adds_drops    = candidate.checkpoint_drops > prior_drops;
                    const bool frees_host_kv = candidate.effect.removed.host.kv_bytes != 0;
                    if (!adds_drops && !frees_host_kv) { continue; }
                    // Stripping a checkpoint (endpoint/rewrite/anchor) makes the
                    // continuation non-resumable: its prefix-index entry disappears and the
                    // next request for the same prefix falls to root. Only dead retained
                    // weight may be stripped here — a live session's state is demoted (a
                    // copy survives) instead, never destroyed.
                    if (adds_drops && victim_live(victim_index)) { continue; }
                }
                const PressureDecision* effective = &candidate;
                PressureDecision trimmed;
                if (!candidate.evicts_continuation && overlaps_committed(victim_index, candidate)) {
                    // Page-disjointness: a host-relief action (DropHostDuplicate on a
                    // device+Host resident victim, or a demote upgrade) whose page range
                    // overlaps another victim's committed pages must be trimmed to its
                    // exclusive pages rather than rejected outright. A live main's
                    // DropHostDuplicate spans the shared prefix (committed by the subs
                    // Phase 1 demoted), so the untrimmed option is always blocked here;
                    // trimming leaves the main's exclusive host copies — the exact
                    // redundant bytes that should pay the Phase 1 host debt.
                    std::optional<PressureDecision> t =
                        trim_to_exclusive(victim_index, candidate);
                    if (!t ||
                        (t->main_kv_changes.empty() && t->backend_kv_changes.empty() &&
                         t->state_changes.empty())) {
                        continue;
                    }
                    trimmed  = std::move(*t);
                    effective = &trimmed;
                }
                const detail::PhysicalResources child =
                    projected_residual(choice_scratch, victim_index, effective);
                const bool dhd = std::any_of(
                    effective->main_kv_changes.begin(), effective->main_kv_changes.end(),
                    [](const auto& a) {
                        return a.kind == PressureKVDecisionKind::DropHostDuplicate &&
                               a.page_count != 0;
                    });
                if (!(residual_key(child) < residual_key(residual))) {
                    if (dhd) {
                        std::fprintf(stderr,
                                     "[P2SKIP] dhd victim=%u%s rem_host=%zu child_host=%zu "
                                     "resid_host=%zu\n",
                                     owners[options.victims[victim_index].owner_index].id.value,
                                     owners[options.victims[victim_index].owner_index].shared ? "s"
                                                                                             : "p",
                                     static_cast<std::size_t>(effective->effect.removed.host.kv_bytes),
                                     static_cast<std::size_t>(child.host.kv_bytes),
                                     static_cast<std::size_t>(residual.host.kv_bytes));
                    }
                    continue;
                }
                if (dhd) {
                    std::fprintf(stderr, "[P2DHD] selected victim=%u%s rem_host=%zu\n",
                                 owners[options.victims[victim_index].owner_index].id.value,
                                 owners[options.victims[victim_index].owner_index].shared ? "s" : "p",
                                 static_cast<std::size_t>(effective->effect.removed.host.kv_bytes));
                }
                selected = Selection{
                    .victim_index = victim_index,
                    .decision     = *effective,
                    .residual     = child,
                };
                break;
            }
            if (selected) { break; }
        }
        if (!selected) { break; }
        pp_selection("P2", *selected);
        choice_scratch[selected->victim_index] = intern_selection(*selected);
        commit_pages(selected->victim_index, selected->decision);
    }

    // Phase 3 — evict: last resort. While still infeasible, evict the smallest
    // victims first (eviction order) whose eviction strictly reduces the total
    // over-commitment. Both tiers are full and everything retainable has been
    // retained at this point. Live victims are skipped: their overflow is demoted
    // (or shed) by Phase 4's host-placement fix-up, which can spend the Host that
    // these dead-weight evictions free; evicting a session here would destroy a
    // prefix a demote could have preserved.
    for (std::size_t step = 0; step < maximum_steps; ++step) {
        const detail::PhysicalResources residual =
            projected_residual(choice_scratch, std::nullopt, nullptr);
        if (feasible(residual)) { break; }
        std::optional<Selection> selected;
        for (const std::size_t victim_index : eviction_order) {
            if (victim_live(victim_index)) { continue; }
            const CandidateVictimOptions& victim = options.victims[victim_index];
            if (victim.eviction_choice == 0 || victim.eviction_choice > victim.decisions.size()) {
                continue;
            }
            if (choice_scratch[victim_index] == victim.eviction_choice) { continue; }
            const PressureDecision& eviction = victim.decisions[victim.eviction_choice - 1U];
            const detail::PhysicalResources child =
                projected_residual(choice_scratch, victim_index, &eviction);
            if (!(residual_key(child) < residual_key(residual))) { continue; }
            selected = Selection{
                .victim_index = victim_index,
                .decision     = eviction,
                .residual     = child,
            };
            break;
        }
        if (!selected) { break; }
        pp_selection("P3", *selected);
        choice_scratch[selected->victim_index] =
            options.victims[selected->victim_index].eviction_choice;
    }

    // Phase 4 — Host-placement verification. The byte model (projected_residual) can
    // declare a plan Host-feasible when the real Host arena cannot place its demotes:
    // free space may be fragmented, and evicting shared victims frees no exclusive Host
    // (their pages are still referenced by the live main's prefix). Ask the real
    // allocator; if the plan's demotes do not fit, run real composition, and if that
    // still fails convert the demoted victim with the largest Host demand into an
    // eviction (drops its whole Host demand and frees more device), re-checking until
    // the plan is genuinely feasible or no demote remains. Converges monotonically and
    // only destroys what the Host tier cannot retain.
    std::vector<std::uint32_t> phase4_main_pages;
    std::vector<std::uint32_t> phase4_back_pages;
    const auto collect_demotes = [&]() {
        phase4_main_pages.clear();
        phase4_back_pages.clear();
        for (std::size_t index = 0; index < options.victims.size(); ++index) {
            const std::uint16_t choice = choice_scratch[index];
            if (choice == 0) { continue; }
            const PressureDecision& decision = options.victims[index].decisions[choice - 1U];
            if (decision.evicts_continuation) { continue; }
            for (const PressureKVDecision& kv : decision.main_kv_changes) {
                if (kv.kind == PressureKVDecisionKind::DemoteToHost && kv.page_count != 0) {
                    phase4_main_pages.push_back(kv.page_count);
                }
            }
            for (const PressureKVDecision& kv : decision.backend_kv_changes) {
                if (kv.kind == PressureKVDecisionKind::DemoteToHost && kv.page_count != 0) {
                    phase4_back_pages.push_back(kv.page_count);
                }
            }
        }
    };
    const auto compose_accepts = [&]() {
        std::vector<const ContinuationHandle*> p_owners;
        std::vector<runtime::PlanningOwnerId> p_ids;
        std::vector<const PressureDecision*> p_decisions;
        std::vector<const SharedPrefixHandle*> s_owners;
        std::vector<runtime::PlanningOwnerId> s_ids;
        std::vector<const PressureDecision*> s_decisions;
        for (std::size_t index = 0; index < options.victims.size(); ++index) {
            const std::uint16_t choice = choice_scratch[index];
            if (choice == 0) { continue; }
            const CandidateVictimOptions& victim = options.victims[index];
            if (victim.owner_index >= owners.size()) {
                throw std::logic_error("pressure planning victim owner is invalid");
            }
            const Owner& owner             = owners[victim.owner_index];
            const PressureDecision* decision = &victim.decisions[choice - 1U];
            if (owner.shared) {
                s_owners.push_back(owner.shared_handle);
                s_ids.push_back(owner.id);
                s_decisions.push_back(decision);
            } else {
                p_owners.push_back(owner.private_handle);
                p_ids.push_back(owner.id);
                p_decisions.push_back(decision);
            }
        }
        const PhysicalCandidateBinding& binding = candidates[selected_candidate];
        if (binding.admission != nullptr) {
            AdmissionCandidate copy(std::make_unique<AdmissionCandidateImpl>(*binding.admission));
            if (!program->compose_pressure_candidate(*copy.impl_, p_owners, p_ids, p_decisions,
                                                     s_owners, s_ids, s_decisions)) {
                return false;
            }
            return copy.impl_->blocked_host_allocation_bytes == 0 &&
                   program->physical_peak_fits_trust_host_allocation(
                       copy.impl_->demand.physical_peak_additional);
        }
        if (binding.capture != nullptr) {
            CapturePressureCandidate copy(
                std::make_unique<CapturePressureCandidateImpl>(*binding.capture));
            if (!program->compose_pressure_candidate(*copy.impl_, p_owners, p_ids, p_decisions,
                                                     s_owners, s_ids, s_decisions)) {
                return false;
            }
            return copy.impl_->blocked_host_allocation_bytes == 0 &&
                   program->physical_peak_fits_trust_host_allocation(
                       copy.impl_->demand.physical_peak_additional);
        }
        return false;
    };
    collect_demotes();
    if (!phase4_main_pages.empty() || !phase4_back_pages.empty()) {
        std::fprintf(stderr, "[PP] P4 host_demand_pages=%zu+%zu\n", phase4_main_pages.size(),
                     phase4_back_pages.size());
        if (!program->host_kv_requests_fit(phase4_main_pages, phase4_back_pages)) {
            std::fprintf(stderr, "[PP] P4 allocator_rejects_demotes\n");
            for (std::size_t step = 0; step < maximum_steps; ++step) {
                if (compose_accepts()) {
                    // compose_accepts credits the plan's own Host releases (DHD /
                    // eviction) against its demote demand. That credit is only real
                    // once materialized, and concurrent in-flight plans can
                    // double-count the same frees against one snapshot. A plan is
                    // acceptable only when its remaining host-adding demotes also fit
                    // the conservative no-release check; otherwise the fix-up below
                    // converts dead host-adding demotes to evictions until the plan
                    // is genuinely materializable.
                    collect_demotes();
                    if (phase4_main_pages.empty() && phase4_back_pages.empty()) { break; }
                    if (program->host_kv_requests_fit(phase4_main_pages, phase4_back_pages)) {
                        break;
                    }
                }
                // Host-placement fix-up. Two complementary actions free what the plan
                // still needs: demote a live victim's overflow to the freed Host (an
                // active session's prefix must never be destroyed while a demote of it
                // could fit), and evict dead retained weight (long-idle contexts) to
                // free Host and device for those demotes. Recency (victim_live) is the
                // live/dead split — the retention weight cannot be trusted here because
                // the live-session escalation never engages on the private-endpoint
                // reuse path, so active sessions would otherwise rank as ordinary
                // RecentPrivate weight and be shed before the dead weight.
                {
                    const detail::PhysicalResources residual =
                        projected_residual(choice_scratch, std::nullopt, nullptr);
                    std::optional<Selection> demote;
                    for (const std::size_t victim_index : eviction_order) {
                        if (!victim_live(victim_index)) { continue; }
                        CandidateVictimOptions& victim = options.victims[victim_index];
                        const std::uint16_t current_choice = choice_scratch[victim_index];
                        const PressureDecision* current =
                            current_choice == 0 ? nullptr
                                                : &victim.decisions[current_choice - 1U];
                        if (current != nullptr && current->evicts_continuation) { continue; }
                        refresh_options(victim, residual, current);
                        for (std::uint16_t choice = 1; choice <= victim.decisions.size();
                             ++choice) {
                            const PressureDecision& candidate = victim.decisions[choice - 1U];
                            if (candidate.evicts_continuation) { continue; }
                            const std::uint32_t prior_drops =
                                current == nullptr ? 0 : current->checkpoint_drops;
                            if (candidate.checkpoint_drops > prior_drops) { continue; }
                            const PressureDecision* effective = &candidate;
                            PressureDecision trimmed;
                            if (overlaps_committed(victim_index, candidate)) {
                                std::optional<PressureDecision> t =
                                    trim_to_exclusive(victim_index, candidate);
                                if (!t ||
                                    (t->main_kv_changes.empty() &&
                                     t->backend_kv_changes.empty() &&
                                     t->state_changes.empty())) {
                                    continue;
                                }
                                trimmed = std::move(*t);
                                effective = &trimmed;
                            }
                            const detail::PhysicalResources child =
                                projected_residual(choice_scratch, victim_index, effective);
                            if (device_key(child) >= device_key(residual)) { continue; }
                            if (child.host.kv_bytes != 0) { continue; }
                            demote = Selection{
                                .victim_index = victim_index,
                                .decision     = *effective,
                                .residual     = child,
                            };
                            break;
                        }
                        if (demote) { break; }
                    }
                    if (demote) {
                        pp_selection("P4D", *demote);
                        choice_scratch[demote->victim_index] = intern_selection(*demote);
                        commit_pages(demote->victim_index, demote->decision);
                        continue;
                    }
                }
                // Evict dead retained weight to free Host and device for the demotes.
                // Live victims are excluded here: they are the last resort and are only
                // shed when no demote could absorb the shortfall. Dead victims that an
                // earlier phase decided as a demote may be upgraded to an eviction —
                // for dead weight that strictly frees more (device and Host) and
                // matches Phase 3's handling of already-decided victims.
                std::optional<std::size_t> evict;
                for (const std::size_t victim_index : eviction_order) {
                    if (victim_live(victim_index)) { continue; }
                    const CandidateVictimOptions& victim = options.victims[victim_index];
                    if (victim.eviction_choice == 0 ||
                        victim.eviction_choice > victim.decisions.size()) {
                        continue;
                    }
                    if (choice_scratch[victim_index] == victim.eviction_choice) { continue; }
                    evict = victim_index;
                    break;
                }
                if (evict) {
                    const Owner& evict_owner = owners[options.victims[*evict].owner_index];
                    std::fprintf(stderr, "[PP] P4 evict=%u%s\n", evict_owner.id.value,
                                 evict_owner.shared ? "s" : "p");
                    choice_scratch[*evict] = options.victims[*evict].eviction_choice;
                    continue;
                }
                // No undecided dead weight left: convert the least-valuable host-adding
                // demote to an eviction (drops its whole Host demand). Live victims are
                // excluded: an actively-stepping session's continuation must never be
                // destroyed to absorb a Host shortfall. When no dead demote can absorb
                // the shortfall the plan stays infeasible and the admission back-pressures
                // instead of shedding a live victim.
                std::optional<std::size_t> convert;
                for (const std::size_t victim_index : eviction_order) {
                    if (victim_live(victim_index)) { continue; }
                    const std::uint16_t choice = choice_scratch[victim_index];
                    if (choice == 0) { continue; }
                    const CandidateVictimOptions& victim = options.victims[victim_index];
                    const PressureDecision& decision = victim.decisions[choice - 1U];
                    if (decision.evicts_continuation) { continue; }
                    if (decision.effect.added.host.kv_bytes == 0) { continue; }
                    if (victim.eviction_choice == 0) { continue; }
                    convert = victim_index;
                    break;
                }
                if (!convert) {
                    // Last resort: shed an undecided dead victim whose eviction choice
                    // was never taken. Live victims are excluded here too — the Host
                    // tier genuinely cannot retain the working set, so the plan is left
                    // infeasible and the admission is blocked instead of destroying an
                    // active session.
                    std::optional<std::size_t> dead_evict;
                    for (const std::size_t victim_index : eviction_order) {
                        if (victim_live(victim_index)) { continue; }
                        if (choice_scratch[victim_index] != 0) { continue; }
                        const CandidateVictimOptions& victim = options.victims[victim_index];
                        if (victim.eviction_choice == 0) { continue; }
                        dead_evict = victim_index;
                        break;
                    }
                    if (dead_evict) {
                        const Owner& evict_owner =
                            owners[options.victims[*dead_evict].owner_index];
                        std::fprintf(stderr, "[PP] P4 dead_evict=%u%s\n", evict_owner.id.value,
                                     evict_owner.shared ? "s" : "p");
                        choice_scratch[*dead_evict] =
                            options.victims[*dead_evict].eviction_choice;
                        continue;
                    }
                    std::fprintf(stderr, "[PP] P4 no_convertible_demote\n");
                    break;
                }
                const Owner& convert_owner = owners[options.victims[*convert].owner_index];
                std::fprintf(stderr, "[PP] P4 convert=%u%s host=%llu\n",
                             convert_owner.id.value, convert_owner.shared ? "s" : "p",
                             static_cast<unsigned long long>(
                                 options.victims[*convert]
                                     .decisions[choice_scratch[*convert] - 1U]
                                     .effect.added.host.kv_bytes));
                choice_scratch[*convert] = options.victims[*convert].eviction_choice;
            }
        }
    }

    std::fprintf(stderr, "[PP] final cand=%u choices=", selected_candidate);
    for (std::size_t index = 0; index < options.victims.size(); ++index) {
        const Owner& owner = owners[options.victims[index].owner_index];
        const auto victim_kv = owner_kv(index);
        std::uint32_t vmain = 0;
        if (victim_kv.first && program->text_kv_addresses) {
            vmain = program->text_kv_addresses->mapped_pages(*victim_kv.first);
        }
        std::fprintf(stderr, "%s%u%s(%u):%u", index == 0 ? "" : ",", owner.id.value,
                     owner.shared ? "s" : "p", vmain, choice_scratch[index]);
    }
    std::fprintf(stderr, "\n");

    TargetNode* existing = find_target(selected_candidate, choice_scratch);
    const std::size_t maximum =
        candidates.size() + 1U + planning_detail::kOptionalTargetCapacity;
    if (existing == nullptr && targets.size() >= maximum) { return std::nullopt; }
    const std::uint32_t target_index =
        existing != nullptr ? static_cast<std::uint32_t>(existing - targets.data())
                            : intern_target(selected_candidate, choice_scratch);
    qwen3_5::PressureTargetHandle handle;
    handle.session_    = this;
    handle.generation_ = generation;
    handle.index_      = target_index;
    return handle;
}

runtime::PressureTargetGuidance
PressurePlanningSessionImpl::guidance(qwen3_5::PressureTargetHandle target) {
    if (!valid(target) || scratch_live) {
        throw std::logic_error("pressure target guidance is stale or conflicts with expansion");
    }
    TargetNode& node = targets[target.index_];
    auto result = guidance_choices(node.candidate_index, victim_choices(node), node.stable_ordinal);
    if (node.assessed_residual &&
        *node.assessed_residual !=
            construction_residual(node.candidate_index, victim_choices(node))) {
        result.physical.requires_exact_feedback = true;
    }
    return result;
}

runtime::PressureTargetGuidance PressurePlanningSessionImpl::guidance_choices(
    std::uint32_t candidate_index, std::span<const std::uint16_t> choices, std::uint32_t ordinal,
    std::optional<std::size_t> override_owner, const PressureDecision* override_decision) {
    populate_options(candidate_index);
    const CandidateState& candidate = *candidates[candidate_index].state;
    const CandidateOptions& options = candidate_options[candidate_index];
    if (choices.size() != options.victims.size()) {
        throw std::logic_error("pressure target victim domain changed");
    }

    guidance_outcomes.clear();
    guidance_checkpoint_changes.clear();
    guidance_recovery.clear();
    bool recovery_complete = true;
    PlanningTransferAccumulator estimated_pressure;
    detail::PhysicalDelta approximate_pressure;
    std::uint32_t total_degradation = 0;
    std::uint32_t total_dropped     = 0;
    for (std::size_t index = 0; index < options.victims.size(); ++index) {
        const std::uint16_t choice                   = choices[index];
        const CandidateVictimOptions& victim_options = options.victims[index];
        if (victim_options.owner_index >= owners.size() ||
            choice > victim_options.decisions.size()) {
            throw std::logic_error("pressure target guidance owner choice is invalid");
        }
        const PressureDecision* selected =
            override_owner && *override_owner == index
                ? override_decision
                : (choice == 0 ? nullptr : &victim_options.decisions[choice - 1U]);
        if (selected == nullptr) { continue; }
        const Owner& victim_owner        = owners[victim_options.owner_index];
        const PressureDecision& decision = *selected;
        for (const auto checkpoint : decision.dropped_checkpoints) {
            guidance_checkpoint_changes.push_back(
                {.owner = victim_owner.id, .checkpoint = checkpoint, .survives = false});
        }
        PlanningTransferAccumulator recovery;
        for (auto transfer : decision.transfer_requirements) {
            if (transfer.direction == runtime::ContextTransferDirection::DeviceToHost) {
                transfer.direction = runtime::ContextTransferDirection::HostToDevice;
                recovery.append(std::span<const runtime::ContextTransferRequirement>(&transfer, 1));
            }
        }
        guidance_recovery.push_back(
            {.owner = victim_owner.id, .additional_restore = recovery.work});
        if (!decision.evicts_continuation && decision.transfer_requirements.empty() &&
            (!decision.state_changes.empty() || !decision.main_kv_changes.empty() ||
             !decision.backend_kv_changes.empty())) {
            recovery_complete = false;
        }
        approximate_pressure.added =
            planning_resource_sum(approximate_pressure.added, decision.effect.added);
        approximate_pressure.removed =
            planning_resource_sum(approximate_pressure.removed, decision.effect.removed);
        estimated_pressure.append(decision.transfer_requirements);
        const std::uint32_t units = degradation_units(decision);
        total_degradation =
            planning_saturating_u32(static_cast<std::uint64_t>(total_degradation) + units);
        total_dropped = planning_saturating_u32(static_cast<std::uint64_t>(total_dropped) +
                                                decision.checkpoint_drops);
        guidance_outcomes.push_back(runtime::PressureOwnerOutcome{
            .owner             = victim_owner.id,
            .disposition       = decision.evicts_continuation ? runtime::VictimDisposition::Evicted
                                                              : runtime::VictimDisposition::Retained,
            .degradation_units = units,
            .dropped_checkpoints = decision.checkpoint_drops,
        });
    }
    detail::PhysicalResources residual =
        program->guided_materialization_deficit(candidate, approximate_pressure);
    residual.host.kv_bytes =
        std::max(residual.host.kv_bytes, candidate.blocked_host_allocation_bytes);

    const detail::PhysicalResources capacity = program->admission_capacity();
    constexpr std::uint64_t kResidualOne     = 1ULL << 20U;
    const auto normalized                    = [](std::uint64_t value, std::uint64_t limit) {
        if (value == 0) { return std::uint64_t{0}; }
        if (limit == 0 || value >= limit) { return kResidualOne; }
        if (value > std::numeric_limits<std::uint64_t>::max() / kResidualOne) {
            return kResidualOne;
        }
        const std::uint64_t scaled = value * kResidualOne;
        return std::max<std::uint64_t>(1, scaled / limit + (scaled % limit != 0 ? 1U : 0U));
    };
    std::uint32_t constraints  = 0;
    std::uint64_t residual_q20 = 0;
    const auto append_residual = [&](std::uint64_t value, std::uint64_t limit) {
        if (value == 0) { return; }
        ++constraints;
        planning_saturating_add(residual_q20, normalized(value, limit));
    };
    append_residual(residual.device.active_lanes, capacity.device.active_lanes);
    append_residual(residual.device.state_slots, capacity.device.state_slots);
    append_residual(residual.device.main_kv_pages, capacity.device.main_kv_pages);
    append_residual(residual.device.backend_kv_pages, capacity.device.backend_kv_pages);
    append_residual(residual.host.state_slots, capacity.host.state_slots);
    append_residual(residual.host.kv_bytes, capacity.host.kv_bytes);

    std::array<std::uint64_t, 6> maximum_additional_relief{};
    const auto update_relief = [&](std::size_t dimension, std::uint64_t eviction_removed,
                                   std::uint64_t eviction_added, std::uint64_t current_removed,
                                   std::uint64_t current_added) {
        const auto saturating_sum = [](std::uint64_t left, std::uint64_t right) {
            return right > std::numeric_limits<std::uint64_t>::max() - left
                       ? std::numeric_limits<std::uint64_t>::max()
                       : left + right;
        };
        const std::uint64_t released         = saturating_sum(eviction_removed, current_added);
        const std::uint64_t consumed         = saturating_sum(eviction_added, current_removed);
        maximum_additional_relief[dimension] = std::max(
            maximum_additional_relief[dimension], released > consumed ? released - consumed : 0U);
    };
    for (std::size_t index = 0; index < options.victims.size(); ++index) {
        const CandidateVictimOptions& victim_options = options.victims[index];
        const std::uint16_t eviction_choice          = victim_options.eviction_choice;
        if (eviction_choice == 0 || choices[index] == eviction_choice) { continue; }
        const PressureDecision& eviction = victim_options.decisions[eviction_choice - 1U];
        const PressureDecision* current =
            choices[index] == 0 ? nullptr : &victim_options.decisions[choices[index] - 1U];
        const detail::PhysicalDelta empty{};
        const detail::PhysicalDelta& prior = current == nullptr ? empty : current->effect;
        update_relief(0, eviction.effect.removed.device.active_lanes,
                      eviction.effect.added.device.active_lanes, prior.removed.device.active_lanes,
                      prior.added.device.active_lanes);
        update_relief(1, eviction.effect.removed.device.state_slots,
                      eviction.effect.added.device.state_slots, prior.removed.device.state_slots,
                      prior.added.device.state_slots);
        update_relief(2, eviction.effect.removed.device.main_kv_pages,
                      eviction.effect.added.device.main_kv_pages,
                      prior.removed.device.main_kv_pages, prior.added.device.main_kv_pages);
        update_relief(3, eviction.effect.removed.device.backend_kv_pages,
                      eviction.effect.added.device.backend_kv_pages,
                      prior.removed.device.backend_kv_pages, prior.added.device.backend_kv_pages);
        update_relief(4, eviction.effect.removed.host.state_slots,
                      eviction.effect.added.host.state_slots, prior.removed.host.state_slots,
                      prior.added.host.state_slots);
        update_relief(5, eviction.effect.removed.host.kv_bytes, eviction.effect.added.host.kv_bytes,
                      prior.removed.host.kv_bytes, prior.added.host.kv_bytes);
    }
    const std::array<std::uint64_t, 6> residual_values{
        residual.device.active_lanes,  residual.device.state_slots,
        residual.device.main_kv_pages, residual.device.backend_kv_pages,
        residual.host.state_slots,     residual.host.kv_bytes,
    };
    std::uint32_t remaining_steps = 0;
    for (std::size_t index = 0; index < residual_values.size(); ++index) {
        if (residual_values[index] == 0) { continue; }
        if (maximum_additional_relief[index] == 0) {
            remaining_steps = std::numeric_limits<std::uint32_t>::max();
            break;
        }
        const std::uint64_t steps =
            1U + (residual_values[index] - 1U) / maximum_additional_relief[index];
        remaining_steps = std::max(remaining_steps, planning_saturating_u32(steps));
    }
    return runtime::PressureTargetGuidance{
        .physical =
            {
                .unsatisfied_constraints   = constraints,
                .estimated_remaining_steps = remaining_steps,
                .normalized_residual_q20   = residual_q20,
                .requires_exact_feedback   = candidate.blocked_host_allocation_bytes != 0,
            },
        .estimated_machine_work     = materialization_machine_work(candidate, estimated_pressure),
        .owner_outcomes             = guidance_outcomes,
        .candidate                  = candidate_ids[candidate_index],
        .stable_target_ordinal      = ordinal,
        .degradation_units          = total_degradation,
        .dropped_checkpoints        = total_dropped,
        .source_mode                = candidate.source_mode,
        .checkpoint_changes         = guidance_checkpoint_changes,
        .recovery_estimates         = guidance_recovery,
        .recovery_estimate_complete = recovery_complete,
    };
}

qwen3_5::AssessedPressureTarget
PressurePlanningSessionImpl::assess(qwen3_5::PressureTargetHandle target) {
    if (!valid(target) || scratch_live) {
        throw std::logic_error("pressure target assessment is stale or conflicts with expansion");
    }
    TargetNode& node = targets[target.index_];
    populate_options(node.candidate_index);
    const CandidateState& candidate              = *candidates[node.candidate_index].state;
    const CandidateOptions& options              = candidate_options[node.candidate_index];
    const std::span<const std::uint16_t> choices = victim_choices(node);
    if (choices.size() != options.victims.size()) {
        throw std::logic_error("pressure target victim domain changed");
    }
    const bool identity_target = std::all_of(choices.begin(), choices.end(),
                                             [](std::uint16_t choice) { return choice == 0; });

    selected_private_owners.clear();
    selected_private_owner_ids.clear();
    selected_private_decisions.clear();
    selected_shared_owners.clear();
    selected_shared_owner_ids.clear();
    selected_shared_decisions.clear();
    recovery_private_owners.clear();
    recovery_private_decisions.clear();
    recovery_private_owner_ids.clear();
    recovery_shared_owners.clear();
    recovery_shared_decisions.clear();
    recovery_shared_owner_ids.clear();
    assessment_outcomes.clear();
    assessment_impact_projections.clear();
    assessment_recovery_alternatives.clear();
    std::fill(projected_owner_decisions.begin(), projected_owner_decisions.end(), nullptr);

    std::uint64_t projection_work   = 1;
    std::uint32_t total_degradation = 0;
    std::uint32_t total_dropped     = 0;
    for (std::size_t index = 0; index < options.victims.size(); ++index) {
        const std::uint16_t choice                   = choices[index];
        const CandidateVictimOptions& victim_options = options.victims[index];
        if (victim_options.owner_index >= owners.size() ||
            choice > victim_options.decisions.size()) {
            throw std::logic_error("pressure target owner choice is invalid");
        }
        const Owner& owner = owners[victim_options.owner_index];
        const PressureDecision* decision =
            choice == 0 ? nullptr : &victim_options.decisions[choice - 1U];
        projected_owner_decisions[victim_options.owner_index] = decision;
        if (decision == nullptr) { continue; }
        if (owner.shared) {
            selected_shared_owners.push_back(owner.shared_handle);
            selected_shared_owner_ids.push_back(owner.id);
            selected_shared_decisions.push_back(decision);
        } else {
            selected_private_owners.push_back(owner.private_handle);
            selected_private_owner_ids.push_back(owner.id);
            selected_private_decisions.push_back(decision);
        }
        const std::uint32_t units   = degradation_units(*decision);
        const std::uint32_t dropped = dropped_checkpoint_count(*decision);
        total_degradation =
            planning_saturating_u32(static_cast<std::uint64_t>(total_degradation) + units);
        total_dropped =
            planning_saturating_u32(static_cast<std::uint64_t>(total_dropped) + dropped);
        assessment_outcomes.push_back(runtime::PressureOwnerOutcome{
            .owner             = owner.id,
            .disposition       = decision->evicts_continuation ? runtime::VictimDisposition::Evicted
                                                               : runtime::VictimDisposition::Retained,
            .degradation_units = units,
            .dropped_checkpoints = dropped,
        });
        ++projection_work;
    }
    for (std::size_t index = 0; index < owners.size(); ++index) {
        const Owner& owner               = owners[index];
        const PressureDecision* decision = projected_owner_decisions[index];
        if (owner.shared) {
            recovery_shared_owners.push_back(owner.shared_handle);
            recovery_shared_decisions.push_back(decision);
            recovery_shared_owner_ids.push_back(owner.id);
        } else {
            recovery_private_owners.push_back(owner.private_handle);
            recovery_private_decisions.push_back(decision);
            recovery_private_owner_ids.push_back(owner.id);
        }
    }

    bool recovery_projection_valid = true;
    if (!identity_target) {
        recovery_projection_valid = program->pressure_checkpoint_recovery_impacts(
            candidate, recovery_private_owners, recovery_private_decisions,
            recovery_private_owner_ids, recovery_shared_owners, recovery_shared_decisions,
            recovery_shared_owner_ids, assessment_impact_projections,
            assessment_recovery_alternatives, recovery_scratch, projection_work);
    }

    runtime::MaterializationPhysicalStatus status =
        runtime::MaterializationPhysicalStatus::StructuralInvalid;
    std::optional<AdmissionCandidate> composed;
    std::optional<CapturePressureCandidate> composed_capture;
    const CandidateState* projected         = &candidate;
    const PhysicalCandidateBinding& binding = candidates[node.candidate_index];
    if (identity_target) {
        status = candidate.identity_assessment.physical_status;
    } else if (recovery_projection_valid) {
        if (binding.admission != nullptr) {
            AdmissionCandidate copy(std::make_unique<AdmissionCandidateImpl>(*binding.admission));
            if (program->compose_pressure_candidate(
                    *copy.impl_, selected_private_owners, selected_private_owner_ids,
                    selected_private_decisions, selected_shared_owners, selected_shared_owner_ids,
                    selected_shared_decisions)) {
                projected = copy.impl_.get();
                composed.emplace(std::move(copy));
            }
        } else {
            CapturePressureCandidate copy(
                std::make_unique<CapturePressureCandidateImpl>(*binding.capture));
            if (program->compose_pressure_candidate(
                    *copy.impl_, selected_private_owners, selected_private_owner_ids,
                    selected_private_decisions, selected_shared_owners, selected_shared_owner_ids,
                    selected_shared_decisions)) {
                projected = copy.impl_.get();
                composed_capture.emplace(std::move(copy));
            }
        }
        if (composed || composed_capture) {
            status = runtime::MaterializationPhysicalStatus::Infeasible;
            if (projected->blocked_host_allocation_bytes == 0 &&
                program->physical_peak_fits_trust_host_allocation(
                    projected->demand.physical_peak_additional)) {
                status = runtime::MaterializationPhysicalStatus::Feasible;
            } else {
                const detail::PhysicalResources occupied = program->physical_occupancy();
                const detail::PhysicalResources limits   = program->admission_capacity();
                const detail::PhysicalResources& peak = projected->demand.physical_peak_additional;
                const auto text_res =
                    program->text_kv_pages != nullptr
                        ? program->text_kv_pages->replica_residency_counts()
                        : qwen3_5::detail::LogicalKVPageStore::ReplicaResidencyCounts{};
                const auto back_res =
                    program->backend_kv_pages != nullptr
                        ? program->backend_kv_pages->replica_residency_counts()
                        : qwen3_5::detail::LogicalKVPageStore::ReplicaResidencyCounts{};
                std::fprintf(
                    stderr,
                    "[AS] INFEASIBLE cand=%u blocked=%zu peak_main=%u occ_main=%u lim_main=%u "
                    "peak_back=%u occ_back=%u lim_back=%u pvt=%zu shd=%zu host_occ=%zu "
                    "host_lim=%zu t(d=%u,h=%u,b=%u) k(d=%u,h=%u,b=%u)\n",
                    node.candidate_index,
                    static_cast<std::size_t>(projected->blocked_host_allocation_bytes),
                    peak.device.main_kv_pages, occupied.device.main_kv_pages,
                    limits.device.main_kv_pages, peak.device.backend_kv_pages,
                    occupied.device.backend_kv_pages, limits.device.backend_kv_pages,
                    selected_private_decisions.size(), selected_shared_decisions.size(),
                    static_cast<std::size_t>(occupied.host.kv_bytes),
                    static_cast<std::size_t>(limits.host.kv_bytes), text_res.device_only,
                    text_res.host_only, text_res.both, back_res.device_only, back_res.host_only,
                    back_res.both);
            }
        }
    }
    const bool composed_valid = composed.has_value() || composed_capture.has_value();
    if (identity_target || composed_valid) {
        node.assessed_residual                = program->materialization_deficit(*projected);
        node.assessed_residual->host.kv_bytes = std::max(node.assessed_residual->host.kv_bytes,
                                                         projected->blocked_host_allocation_bytes);
    } else {
        node.assessed_residual.reset();
    }
    const runtime::MaterializationMachineWork machine_work =
        identity_target ? candidate.identity_assessment.machine_work
                        : materialization_machine_work(*projected, selected_private_decisions,
                                                       selected_shared_decisions);

    bool expandable = identity_target || (recovery_projection_valid && composed_valid);
    if (expandable) {
        expandable = false;
        for (std::size_t index = 0; index < options.victims.size(); ++index) {
            const CandidateVictimOptions& victim_options = options.victims[index];
            const std::uint16_t choice                   = choices[index];
            if ((choice == 0 && !victim_options.decisions.empty()) ||
                (choice != 0 && choice <= victim_options.decisions.size() &&
                 !victim_options.decisions[choice - 1U].evicts_continuation)) {
                expandable = true;
                break;
            }
        }
    }

    std::uint64_t digest = candidate.identity_assessment.assessment_digest;
    if (!identity_target) {
        digest = 1469598103934665603ULL;
        planning_detail::hash_mix(digest, node.candidate_index);
        planning_detail::hash_mix(digest, node.stable_ordinal);
        planning_detail::hash_mix(digest, static_cast<std::uint8_t>(status));
        planning_detail::hash_mix(digest, machine_work.remaining_prefill_work.tokens);
        for (const TransferWork transfer : machine_work.pressure_transfers) {
            planning_detail::hash_mix(digest, transfer.payload_bytes);
            planning_detail::hash_mix(digest, transfer.copy_operations);
        }
        for (const TransferWork transfer : machine_work.candidate_transfers) {
            planning_detail::hash_mix(digest, transfer.payload_bytes);
            planning_detail::hash_mix(digest, transfer.copy_operations);
        }
        planning_detail::hash_mix(digest, total_degradation);
        for (const std::uint16_t choice : choices) { planning_detail::hash_mix(digest, choice); }
    }

    runtime::PressureTargetAssessment result{
        .physical_status       = status,
        .source_mode           = projected->source_mode,
        .machine_work          = machine_work,
        .owner_outcomes        = assessment_outcomes,
        .checkpoint_impacts    = {},
        .candidate             = candidate_ids[node.candidate_index],
        .stable_target_ordinal = node.stable_ordinal,
        .degradation_units     = total_degradation,
        .dropped_checkpoints   = total_dropped,
        .projection_work       = projection_work,
        .assessment_digest     = digest,
        .expandable            = expandable,
        .root_maximal          = node.root_maximal,
    };
    std::optional<AdmissionCandidate> executable;
    std::optional<CapturePressureCandidate> capture_executable;
    if (status == runtime::MaterializationPhysicalStatus::Feasible) {
        if (identity_target) {
            if (binding.admission != nullptr) {
                executable.emplace(std::make_unique<AdmissionCandidateImpl>(*binding.admission));
            } else {
                capture_executable.emplace(
                    std::make_unique<CapturePressureCandidateImpl>(*binding.capture));
            }
        } else if (composed) {
            executable.emplace(std::move(*composed));
        } else if (composed_capture) {
            capture_executable.emplace(std::move(*composed_capture));
        }
    }
    const std::uint32_t slot_index = acquire_assessment_slot();
    AssessmentSlot& slot           = assessment_slots[slot_index];
    if (slot.owner_outcomes.capacity() < assessment_outcomes.size() ||
        slot.checkpoint_impacts.capacity() < assessment_impact_projections.size() ||
        slot.recovery_alternatives.capacity() < assessment_recovery_alternatives.size()) {
        release_assessment_slot(this, slot_index, slot.generation);
        throw std::logic_error("pressure assessment exceeded preallocated result storage");
    }
    slot.owner_outcomes.insert(slot.owner_outcomes.end(), assessment_outcomes.begin(),
                               assessment_outcomes.end());
    slot.recovery_alternatives.insert(slot.recovery_alternatives.end(),
                                      assessment_recovery_alternatives.begin(),
                                      assessment_recovery_alternatives.end());
    for (const PressureCheckpointRecoveryProjection& impact : assessment_impact_projections) {
        if (impact.alternative_offset > slot.recovery_alternatives.size() ||
            impact.alternative_count >
                slot.recovery_alternatives.size() - impact.alternative_offset) {
            release_assessment_slot(this, slot_index, slot.generation);
            throw std::logic_error("pressure recovery work span is invalid");
        }
        slot.checkpoint_impacts.push_back(runtime::PressureCheckpointRecoveryImpact{
            .owner      = impact.owner,
            .checkpoint = impact.checkpoint,
            .target_recovery_work =
                std::span<const runtime::CheckpointRecoveryAlternativeWork>(
                    slot.recovery_alternatives)
                    .subspan(impact.alternative_offset, impact.alternative_count),
            .survives = impact.survives,
        });
    }
    result.owner_outcomes     = slot.owner_outcomes;
    result.checkpoint_impacts = slot.checkpoint_impacts;
    return qwen3_5::AssessedPressureTarget(this, generation, target.index_, result, slot_index,
                                           slot.generation,
                                           &PressurePlanningSessionImpl::release_assessment_slot,
                                           std::move(executable), std::move(capture_executable));
}

qwen3_5::PreparedPressureExpansion
PressurePlanningSessionImpl::prepare_expansion(qwen3_5::PressureTargetHandle parent,
                                               std::uint32_t maximum_owners) {
    if (!valid(parent) || scratch_live || maximum_owners == 0) {
        throw std::logic_error("pressure expansion parent is stale or scratch is busy");
    }
    const TargetNode& node = targets[parent.index_];
    populate_options(node.candidate_index);
    CandidateOptions& options                           = candidate_options[node.candidate_index];
    const std::span<const std::uint16_t> parent_choices = victim_choices(node);
    if (parent_choices.size() != options.victims.size()) {
        throw std::logic_error("pressure target victim domain changed");
    }
    expansion_scratch.clear();
    prepared_owner_decisions.clear();
    prepared_new_count  = 0;
    scratch_choice_mark = target_choice_arena.size();

    const CandidateState& candidate = *candidates[node.candidate_index].state;
    const std::optional<Core::MaterializationSourceProtection> protection =
        program->materialization_source_protection(candidate);
    if (!protection) { throw std::logic_error("pressure expansion source protection is stale"); }
    const bool identity = std::all_of(parent_choices.begin(), parent_choices.end(),
                                      [](std::uint16_t choice) { return choice == 0; });
    detail::PhysicalResources residual;
    if (identity) {
        residual = candidate.identity_pressure_deficit;
        residual.host.kv_bytes =
            std::max(residual.host.kv_bytes, candidate.blocked_host_allocation_bytes);
    } else {
        if (!node.assessed_residual) {
            throw std::logic_error("pressure target must be assessed before expansion");
        }
        residual = *node.assessed_residual;
    }

    const auto append = [&](std::size_t victim_index, std::uint16_t choice) {
        if (parent_choices.size() > target_choice_arena.capacity() - target_choice_arena.size() ||
            target_choice_arena.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("pressure expansion choice arena is full");
        }
        const std::uint32_t offset = static_cast<std::uint32_t>(target_choice_arena.size());
        target_choice_arena.insert(target_choice_arena.end(), parent_choices.begin(),
                                   parent_choices.end());
        target_choice_arena[offset + victim_index] = choice;
        TargetNode child{
            .candidate_index      = node.candidate_index,
            .victim_choice_offset = offset,
            .victim_choice_count  = static_cast<std::uint32_t>(parent_choices.size()),
        };
        const std::span<const std::uint16_t> child_choices = victim_choices(child);
        const bool duplicate_scratch                       = std::any_of(
            expansion_scratch.begin(), expansion_scratch.end(), [&](const TargetNode& existing) {
                return existing.candidate_index == child.candidate_index &&
                       planning_detail::same_choices(victim_choices(existing), child_choices);
            });
        if (duplicate_scratch) {
            target_choice_arena.resize(offset);
            return;
        }
        const bool existing = find_target(child.candidate_index, child_choices) != nullptr;
        if (!existing) { ++prepared_new_count; }
        expansion_scratch.push_back(child);
    };

    const auto intern_prepared_decision = [&](std::size_t victim_index, PressureDecision decision) {
        std::vector<PressureDecision>& decisions = options.victims[victim_index].decisions;
        const auto existing = std::find(decisions.begin(), decisions.end(), decision);
        if (existing != decisions.end()) {
            return static_cast<std::uint16_t>(1U + (existing - decisions.begin()));
        }
        const auto prepared =
            std::find_if(prepared_owner_decisions.begin(), prepared_owner_decisions.end(),
                         [&](const PreparedOwnerDecision& item) {
                             return item.candidate_index == node.candidate_index &&
                                    item.victim_index == victim_index && item.decision == decision;
                         });
        if (prepared != prepared_owner_decisions.end()) { return prepared->choice; }
        const std::size_t staged = static_cast<std::size_t>(
            std::count_if(prepared_owner_decisions.begin(), prepared_owner_decisions.end(),
                          [&](const PreparedOwnerDecision& item) {
                              return item.candidate_index == node.candidate_index &&
                                     item.victim_index == victim_index;
                          }));
        const std::size_t value = decisions.size() + staged + 1U;
        if (value > std::numeric_limits<std::uint16_t>::max()) {
            throw std::overflow_error("pressure owner target count is not representable");
        }
        const std::uint16_t choice = static_cast<std::uint16_t>(value);
        prepared_owner_decisions.push_back(PreparedOwnerDecision{
            .candidate_index = node.candidate_index,
            .victim_index    = static_cast<std::uint32_t>(victim_index),
            .choice          = choice,
            .decision        = std::move(decision),
        });
        return choice;
    };

    prepared_owner_end = static_cast<std::uint32_t>(std::min<std::size_t>(
        options.victims.size(),
        static_cast<std::size_t>(node.next_expansion_owner) + maximum_owners));
    try {
        for (std::size_t victim_index = node.next_expansion_owner;
             victim_index < prepared_owner_end; ++victim_index) {
            CandidateVictimOptions& victim_options   = options.victims[victim_index];
            const std::uint16_t current_choice       = parent_choices[victim_index];
            std::vector<PressureDecision>& decisions = victim_options.decisions;
            if (current_choice > decisions.size() ||
                (current_choice != 0 && decisions[current_choice - 1U].evicts_continuation)) {
                continue;
            }
            const PressureDecision* current =
                current_choice == 0 ? nullptr : &decisions[current_choice - 1U];
            std::vector<PressureDecision> successors =
                pressure_successors(victim_options, residual, *protection, current);
            for (PressureDecision& successor : successors) {
                const std::uint16_t choice =
                    intern_prepared_decision(victim_index, std::move(successor));
                if (choice == current_choice) { continue; }
                append(victim_index, choice);
            }
        }
    } catch (...) {
        target_choice_arena.resize(scratch_choice_mark);
        expansion_scratch.clear();
        prepared_owner_decisions.clear();
        prepared_new_count = 0;
        throw;
    }

    if (++scratch_generation == 0) { ++scratch_generation; }
    scratch_live = true;
    return qwen3_5::PreparedPressureExpansion(this, generation, scratch_generation, parent.index_,
                                              prepared_new_count);
}

qwen3_5::PressureExpansionView
PressurePlanningSessionImpl::commit_expansion(qwen3_5::PreparedPressureExpansion&& prepared) {
    if (!scratch_live || prepared.session_ != this || prepared.session_generation_ != generation ||
        prepared.scratch_generation_ != scratch_generation ||
        prepared.new_canonical_count_ != prepared_new_count ||
        prepared.parent_index_ >= targets.size()) {
        throw std::logic_error("prepared pressure expansion is stale");
    }
    const std::size_t maximum = candidates.size() + 1U + planning_detail::kOptionalTargetCapacity;
    if (prepared_new_count > maximum - std::min(maximum, targets.size())) {
        throw std::length_error("prepared pressure expansion exceeds the target arena");
    }

    committed_children.clear();
    committed_children.reserve(expansion_scratch.size());
    for (PreparedOwnerDecision& prepared_decision : prepared_owner_decisions) {
        if (prepared_decision.candidate_index >= candidate_options.size()) {
            throw std::logic_error("prepared pressure owner candidate is invalid");
        }
        CandidateOptions& options = candidate_options[prepared_decision.candidate_index];
        if (prepared_decision.victim_index >= options.victims.size()) {
            throw std::logic_error("prepared pressure owner index is invalid");
        }
        CandidateVictimOptions& victim_options   = options.victims[prepared_decision.victim_index];
        std::vector<PressureDecision>& decisions = victim_options.decisions;
        if (prepared_decision.choice != decisions.size() + 1U) {
            throw std::logic_error("prepared pressure owner choice is not canonical");
        }
        decisions.push_back(std::move(prepared_decision.decision));
    }
    std::size_t choice_write          = scratch_choice_mark;
    std::uint32_t committed_new_count = 0;
    for (TargetNode& child : expansion_scratch) {
        const std::span<const std::uint16_t> child_choices = victim_choices(child);
        TargetNode* existing = find_target(child.candidate_index, child_choices);
        std::uint32_t index  = 0;
        if (existing == nullptr) {
            if (child_choices.size() > target_choice_arena.size() - choice_write ||
                targets.size() == targets.capacity()) {
                throw std::length_error("committed pressure target exceeds its arena");
            }
            for (std::size_t choice = 0; choice < child_choices.size(); ++choice) {
                target_choice_arena[choice_write + choice] = child_choices[choice];
            }
            child.victim_choice_offset = static_cast<std::uint32_t>(choice_write);
            choice_write += child_choices.size();
            child.stable_ordinal = static_cast<std::uint32_t>(targets.size());
            targets.push_back(child);
            index = static_cast<std::uint32_t>(targets.size() - 1U);
            index_target(index);
            ++committed_new_count;
        } else {
            index = static_cast<std::uint32_t>(existing - targets.data());
        }
        qwen3_5::PressureTargetHandle handle;
        handle.session_    = this;
        handle.generation_ = generation;
        handle.index_      = index;
        committed_children.push_back(handle);
    }
    if (committed_new_count != prepared_new_count) {
        throw std::logic_error("prepared pressure target count changed before commit");
    }
    target_choice_arena.resize(choice_write);
    const std::uint32_t new_count = prepared_new_count;
    auto& parent                  = targets[prepared.parent_index_];
    parent.next_expansion_owner   = prepared_owner_end;
    const bool complete =
        prepared_owner_end == candidate_options[parent.candidate_index].victims.size();
    prepared.session_            = nullptr;
    prepared.session_generation_ = 0;
    prepared.scratch_generation_ = 0;
    expansion_scratch.clear();
    prepared_owner_decisions.clear();
    prepared_new_count  = 0;
    scratch_choice_mark = target_choice_arena.size();
    scratch_live        = false;
    return qwen3_5::PressureExpansionView{
        .children            = committed_children,
        .new_canonical_count = new_count,
        .complete            = complete,
    };
}

void PressurePlanningSessionImpl::discard_expansion(
    qwen3_5::PreparedPressureExpansion&& prepared) noexcept {
    if (scratch_live && prepared.session_ == this && prepared.session_generation_ == generation &&
        prepared.scratch_generation_ == scratch_generation) {
        expansion_scratch.clear();
        prepared_owner_decisions.clear();
        target_choice_arena.resize(scratch_choice_mark);
        prepared_new_count = 0;
        scratch_live       = false;
    }
    prepared.session_            = nullptr;
    prepared.session_generation_ = 0;
    prepared.scratch_generation_ = 0;
}

runtime::PrefillWork PressurePlanningSessionImpl::shared_capture_split_prefill_work(
    const qwen3_5::AssessedPressureTarget& assessed, const PreparedPromptData& prompt,
    std::span<const std::uint32_t> frontiers) const {
    if (assessed.session_ != this || assessed.session_generation_ != generation || scratch_live ||
        assessed.target_index_ >= targets.size() || !assessed.executable_ ||
        assessed.capture_executable_ ||
        assessed.assessment_.physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        throw std::logic_error("shared capture cost requires a feasible assessed materialization");
    }
    return program->shared_capture_split_prefill_work(*assessed.executable_, prompt, frontiers);
}

std::optional<PressurePlanningSessionImpl::AdmissionCandidate>
PressurePlanningSessionImpl::seal(qwen3_5::AssessedPressureTarget&& assessed,
                                  const PreparedPromptData& prompt,
                                  runtime::FinalScheduleIntent intent) {
    if (assessed.session_ != this || assessed.session_generation_ != generation || scratch_live ||
        assessed.target_index_ >= targets.size() || !assessed.executable_ ||
        assessed.capture_executable_ ||
        assessed.assessment_.physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        throw std::logic_error("pressure assessment is not sealable as materialization");
    }
    std::optional<AdmissionCandidate> sealed = std::move(assessed.executable_);
    assessed.reset();
    program->select_shared_captures(*sealed, prompt, intent.shared_capture_frontiers);
    if (sealed->impl_->blocked_host_allocation_bytes != 0 ||
        program->revalidate_materialization(*sealed, prompt) != runtime::PreflightStatus::Ready) {
        return std::nullopt;
    }
    return sealed;
}

std::optional<PressurePlanningSessionImpl::CapturePressureCandidate>
PressurePlanningSessionImpl::seal_capture(qwen3_5::AssessedPressureTarget&& assessed) {
    if (assessed.session_ != this || assessed.session_generation_ != generation || scratch_live ||
        assessed.target_index_ >= targets.size() || assessed.executable_ ||
        !assessed.capture_executable_ ||
        assessed.assessment_.physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        throw std::logic_error("pressure assessment is not sealable as capture");
    }
    std::optional<CapturePressureCandidate> sealed = std::move(assessed.capture_executable_);
    assessed.reset();
    if (sealed->impl_->blocked_host_allocation_bytes != 0 ||
        !program->physical_peak_fits_trust_host_allocation(
            sealed->impl_->demand.physical_peak_additional)) {
        return std::nullopt;
    }
    return sealed;
}

} // namespace ninfer::models::qwen3_5::detail
