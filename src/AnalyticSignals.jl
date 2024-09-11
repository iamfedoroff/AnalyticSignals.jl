module AnalyticSignals

import CUDA
import FFTW

export rsig2asig!, rsig2aspec!, rspec2aspec!, rspec2asig!,
       asig2rsig!, asig2rspec!, aspec2rspec!, aspec2rsig!


macro krun(ex...)
    N = ex[1]
    call = ex[2]

    args = call.args[2:end]

    @gensym kernel config threads blocks
    code = quote
        local $kernel = CUDA.@cuda launch=false $call
        local $config = CUDA.launch_configuration($kernel.fun)
        local $threads = min($N, $config.threads)
        local $blocks = cld($N, $threads)
        $kernel($(args...); threads=$threads, blocks=$blocks)
    end

    return esc(code)
end


half(N) = iseven(N) ? div(N, 2) : div(N + 1, 2)


# ******************************************************************************
# real signal -> analytic signal
# ******************************************************************************
function rsig2asig!(E::AbstractArray{<:Complex, 1})
    FFTW.ifft!(E)   # time -> frequency [exp(-i*w*t)]
    rspec2aspec!(E)
    FFTW.fft!(E)   # frequency -> time [exp(-i*w*t)]
    return nothing
end


function rsig2asig!(E::AbstractArray{<:Complex, 2})
    FFTW.ifft!(E, [2])   # time -> frequency [exp(-i*w*t)]
    rspec2aspec!(E)
    FFTW.fft!(E, [2])   # frequency -> time [exp(-i*w*t)]
    return nothing
end


function rsig2asig!(E::AbstractArray{<:Complex, 3})
    FFTW.ifft!(E, [3])   # time -> frequency [exp(-i*w*t)]
    rspec2aspec!(E)
    FFTW.fft!(E, [3])   # frequency -> time [exp(-i*w*t)]
    return nothing
end


function rsig2asig!(E::AbstractArray{<:Complex}, FT::FFTW.Plan)
    FT \ E   # time -> frequency [exp(-i*w*t)]
    rspec2aspec!(E)
    FT * E   # frequency -> time [exp(-i*w*t)]
    return nothing
end


# ******************************************************************************
# real signal -> analytic spectrum
# ******************************************************************************
function rsig2aspec!(E::AbstractArray{<:Complex}, FT::FFTW.Plan)
    FT \ E   # time -> frequency [exp(-i*w*t)]
    rspec2aspec!(E)
    return nothing
end


# ******************************************************************************
# real spectrum -> analytic spectrum
#
#         { 2 * Sr(f),  f > 0
# Sa(f) = { Sr(f),      f = 0   = [1 + sgn(f)] * Sr(f)
#         { 0,          f < 0
# ******************************************************************************
function rspec2aspec!(S::AbstractArray{<:Complex, 1})
    N = length(S)
    Nhalf = half(N)
    # S[1] = S[1]   # f = 0
    for i=2:Nhalf
        S[i] = 2 * S[i]   # f > 0
    end
    for i=Nhalf+1:N
        S[i] = 0   # f < 0
    end
    return nothing
end


function rspec2aspec!(S::AbstractArray{<:Complex, 2})
    N1, N2 = size(S)
    for i=1:N1
        @views rspec2aspec!(S[i, :])
    end
    return nothing
end


function rspec2aspec!(S::AbstractArray{<:Complex, 3})
    N1, N2, N3 = size(S)
    for j=1:N2, i=1:N1
        @views rspec2aspec!(S[i, j, :])
    end
    return nothing
end


function rspec2aspec!(S::CUDA.CuArray{<:Complex})
    @krun length(S) _rspec2aspec_kernel!(S)
    return nothing
end


function _rspec2aspec_kernel!(S::CUDA.CuDeviceArray{<:Complex, 1})
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    Nt = length(S)
    Nthalf = half(Nt)
    for it=id:stride:Nt
        if (it >= 2) & (it <= Nthalf)
            S[it] = 2 * S[it]
        end
        if (it >= Nthalf + 1) & (it <= Nt)
            S[it] = 0
        end
    end
    return nothing
end


function _rspec2aspec_kernel!(S::CUDA.CuDeviceArray{<:Complex, 2})
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    Nr, Nt = size(S)
    Nthalf = half(Nt)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        ir = cartesian[k][1]
        it = cartesian[k][2]

        if (it >= 2) & (it <= Nthalf)
            S[ir, it] = 2 * S[ir, it]
        end
        if (it >= Nthalf + 1) & (it <= Nt)
            S[ir, it] = 0
        end
    end
    return nothing
end


function _rspec2aspec_kernel!(S::CUDA.CuDeviceArray{<:Complex, 3})
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    Nx, Ny, Nt = size(S)
    Nthalf = half(Nt)
    cartesian = CartesianIndices((Nx, Ny, Nt))
    for k=id:stride:Nx*Ny*Nt
        ix = cartesian[k][1]
        iy = cartesian[k][2]
        it = cartesian[k][3]

        if (it >= 2) & (it <= Nthalf)
            S[ix, iy, it] = 2 * S[ix, iy, it]
        end
        if (it >= Nthalf + 1) & (it <= Nt)
            S[ix, iy, it] = 0
        end
    end
    return nothing
end


# ******************************************************************************
# real spectrum -> analytic signal
# ******************************************************************************
function rspec2asig!(S::AbstractArray{<:Complex}, FT::FFTW.Plan)
    rspec2aspec!(S)
    FT * S   # frequency -> time [exp(-i*w*t)]
    return nothing
end


# ******************************************************************************
# analytic signal -> real signal
# ******************************************************************************
function asig2rsig!(E::AbstractArray{<:Complex})
    @. E = real(E)
    return nothing
end


# ******************************************************************************
# analytic signal -> real spectrum
# ******************************************************************************
function asig2rspec!(E::AbstractArray{<:Complex}, FT::FFTW.Plan)
    FT \ E   # time -> frequency [exp(-i*w*t)]
    aspec2rspec!(E)
    return nothing
end


function asig2rspec!(
    Sr::A, E::A, FT::FFTW.Plan,
) where A<:AbstractArray{<:Complex}
    FT \ E   # time -> frequency [exp(-i*w*t)]
    aspec2rspec!(Sr, E)
    return nothing
end


# ******************************************************************************
# analytic spectrum -> real spectrum
#
#         { Sa(f) / 2,         f > 0
# Sr(f) = { Sa(f),             f = 0   = [Sa(f) + conj(Sa(-f))] / 2
#         { conj(Sa(-f)) / 2,  f < 0
# ******************************************************************************
function aspec2rspec!(S::AbstractArray{<:Complex, 1})
    N = length(S)
    Nhalf = half(N)
    # S[1] = S[1]   # f = 0
    for i=2:Nhalf
        S[i] = S[i] / 2   # f > 0
    end
    for i=Nhalf+1:N
        S[i] = conj(S[N - i + 2])   # f < 0
    end
    return nothing
end


function aspec2rspec!(S::AbstractArray{<:Complex, 2})
    N1, N2 = size(S)
    for i=1:N1
        @views aspec2rspec!(S[i, :])
    end
    return nothing
end


function aspec2rspec!(S::AbstractArray{<:Complex, 3})
    N1, N2, N3 = size(S)
    for j=1:N2, i=1:N1
        @views aspec2rspec!(S[i, j, :])
    end
    return nothing
end


function aspec2rspec!(Sr::A, Sa::A) where A<:AbstractArray{<:Complex, 1}
    N = length(Sr)
    Sr[1] = Sa[1]   # f = 0
    for i=2:N
        Sr[i] = Sa[i] / 2   # f > 0
    end
    return nothing
end


function aspec2rspec!(Sr::A, Sa::A) where A<:AbstractArray{<:Complex, 2}
    N1, N2 = size(Sr)
    for i=1:N1
        @views aspec2rspec!(Sr[i, :], Sa[i, :])
    end
    return nothing
end


function aspec2rspec!(Sr::A, Sa::A) where A<:AbstractArray{<:Complex, 3}
    N1, N2, N3 = size(Sr)
    for j=1:N2, i=1:N1
        @views aspec2rspec!(Sr[i, j, :], Sa[i, j, :])
    end
    return nothing
end


function aspec2rspec!(Sr::A, Sa::A) where A<:CUDA.CuArray{<:Complex}
    @krun length(Sr) _aspec2rspec_kernel!(Sr, Sa)
    return nothing
end


function _aspec2rspec_kernel!(
    Sr::A, Sa::A,
) where A<:CUDA.CuDeviceArray{<:Complex, 1}
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    Nw = length(Sr)
    for iw=id:stride:Nw
        if iw == 1
            Sr[iw] = Sa[iw]
        else
            Sr[iw] = Sa[iw] / 2
        end
    end
    return nothing
end


function _aspec2rspec_kernel!(
    Sr::A, Sa::A,
) where A<:CUDA.CuDeviceArray{<:Complex, 2}
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    Nr, Nw = size(Sr)
    cartesian = CartesianIndices((Nr, Nw))
    for k=id:stride:Nr*Nw
        i = cartesian[k][1]
        j = cartesian[k][2]
        if j == 1
            Sr[i, j] = Sa[i, j]
        else
            Sr[i, j] = Sa[i, j] / 2
        end
    end
    return nothing
end


function _aspec2rspec_kernel!(
    Sr::A, Sa::A,
) where A<:CUDA.CuDeviceArray{<:Complex, 3}
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    Nx, Ny, Nw = size(Sr)
    cartesian = CartesianIndices((Nx, Ny, Nw))
    for k=id:stride:Nx*Ny*Nw
        ix = cartesian[k][1]
        iy = cartesian[k][2]
        iw = cartesian[k][3]
        if iw == 1
            Sr[ix, iy, iw] = Sa[ix, iy, iw]
        else
            Sr[ix, iy, iw] = Sa[ix, iy, iw] / 2
        end
    end
    return nothing
end


# ******************************************************************************
# analytic spectrum -> real signal
# ******************************************************************************
function aspec2rsig!(S::AbstractArray{<:Complex}, FT::FFTW.Plan)
    aspec2rspec!(S)
    FT * S   # frequency -> time [exp(-i*w*t)]
    return nothing
end


end
