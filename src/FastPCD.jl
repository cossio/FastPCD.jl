module FastPCD

using RestrictedBoltzmannMachines: RBM, Binary, Spin, Potts, Gaussian, ReLU, dReLU
using Optimisers: AbstractRule, setup, update!, Adam

"""
    fpcd!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence, with fast weights.
See http://dl.acm.org/citation.cfm?id=1553374.1553506.
"""
function fast_pcd!(rbm::RBM, data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optim::AbstractRule = Adam(1f-3),
    wts::Union{AbstractVector,Nothing} = nothing,
    steps::Int = 1,
    optimfast::AbstractRule = Adam(1f-2), # optimizer algorithm for fast parameters (should have higher learning rate)
    decayfast::Real = 19/20  # weight decay of fast parameters
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    isnothing(wts) || @assert size(data)[end] == length(wts)

    moments = moments_from_samples(rbm.visible, data; wts)
    vm = sample_from_inputs(rbm.visible, falses(size(rbm.visible)..., batchsize))

    rbmfast = deepcopy(rbm) # store fast parameters

    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w)
    state = setup(optim, ps)

    ps_fast = (; visible = rbmfast.visible.par, hidden = rbmfast.hidden.par, w = rbmfast.w)
    state_fast = setup(optimfast, ps_fast)

    # (Actually, the parameters of rbmfast are the sums θ_regular + θ_fast)
    for _ in 1:epochs, (vd, wd) in minibatches(data, wts; batchsize)
        vm = sample_v_from_v(rbmfast, vm; steps)

        ∂d = ∂free_energy(rbm, vd; wts=wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        gs = (; visible=∂.visible, hidden=∂.hidden, w=∂.w)
        state, ps = update!(state, ps, gs)
        state_fast, ps_fast = update!(state_fast, ps_fast, gs)

        decayfast!(rbmfast, rbm; decay=decayfast)
    end
    return rbm
end

#= Decays parameters of `rbmfast` towards those of `rbm`.
In other words, writing the parametrs of `rbmfast` as
ω_regular + ω_fast, where ω_regular are the prameters of
`rbm`, then here we decay the ω_fast part towards zero. =#
function decayfast!(rbmfast::M, rbm::M; decay::Real) where {M<:RBM}
    decayfast!(rbmfast.visible, rbm.visible; decay)
    decayfast!(rbmfast.hidden, rbm.hidden; decay)
    decayfast!(rbmfast.w, rbm.w; decay)
end

function decayfast!(fast::L, regular::L; decay::Real) where {L<:Union{Binary,Spin,Potts}}
    @assert size(fast) == size(regular)
    decayfast!(fast.θ, regular.θ; decay)
end

function decayfast!(fast::L, regular::L; decay::Real) where {L<:Union{Gaussian,ReLU}}
    @assert size(fast) == size(regular)
    decayfast!(fast.θ, regular.θ; decay)
    decayfast!(fast.γ, regular.γ; decay)
end

function decayfast!(fast::L, regular::L; decay::Real) where {L<:dReLU}
    @assert size(fast) == size(regular)
    decayfast!(fast.θp, regular.θp; decay)
    decayfast!(fast.θn, regular.θn; decay)
    decayfast!(fast.γp, regular.γp; decay)
    decayfast!(fast.γn, regular.γn; decay)
end

function decayfast!(ωfast::AbstractArray, ωregular::AbstractArray; decay::Real)
    @assert size(ωfast) == size(ωregular)
    ωfast .= decay .* ωfast + (1 - decay) .* ωregular
end


end # module
