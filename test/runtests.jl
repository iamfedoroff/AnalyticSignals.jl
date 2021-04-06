using AnalyticSignals
using CUDA
using FFTW
using Test

CUDA.allowscalar(false)

Adc = 0.1

A01 = 1
tau01 = 1
w01 = 10

A02 = 0.5
tau02 = 2 * tau01
w02 = 2 * w01
phi = pi / 2

Nt = 256
t = range(-10 * tau01, 10 * tau01, length=Nt)
Nw = length(FFTW.rfftfreq(Nt))

Er = @. Adc +
     A01 * exp(-0.5 * t^2 / tau01^2) * cos(w01 * t) +
     A02 * exp(-0.5 * t^2 / tau02^2) * cos(w02 * t + phi) + 0im

Ea = @. Adc +
     A01 * exp(-0.5 * t^2 / tau01^2) * exp(-1im * w01 * t) +
     A02 * exp(-0.5 * t^2 / tau02^2) * exp(-1im * (w02 * t + phi))

Sr = FFTW.ifft(Er)
Sa = FFTW.ifft(Ea)


Nr = 512
Er2 = zeros(ComplexF64, (Nr, Nt))
Ea2 = zeros(ComplexF64, (Nr, Nt))
for i=1:Nr
    @. Er2[i, :] = Er
    @. Ea2[i, :] = Ea
end

Sr2 = FFTW.ifft(Er2, [2])
Sa2 = FFTW.ifft(Ea2, [2])


Er2gpu = CUDA.CuArray(Er2)
Ea2gpu = CUDA.CuArray(Ea2)

Sr2gpu = FFTW.ifft(Er2gpu, [2])
Sa2gpu = FFTW.ifft(Ea2gpu, [2])


# ******************************************************************************
# 1D CPU
# ******************************************************************************
# real signal -> analytic signal:
E = copy(Er)
plan = FFTW.plan_fft!(E)
rsig2asig!(E, plan)
@test isapprox(E, Ea, rtol=1e-6)

# real signal -> analytic spectrum:
E = copy(Er)
plan = FFTW.plan_fft!(E)
rsig2aspec!(E, plan)
@test isapprox(E, Sa, rtol=1e-6)

# real spectrum -> analytic spectrum:
S = copy(Sr)
rspec2aspec!(S)
@test isapprox(S, Sa, rtol=1e-6)

# real spectrum -> analytic signal:
# nothing


# ------------------------------------------------------------------------------
# analytic signal -> real signal
E = copy(Ea)
asig2rsig!(E)
@test isapprox(E, Er, rtol=1e-6)

# analytic signal -> real spectrum
# nothing

# analytic spectrum -> real spectrum
S = copy(Sa)
aspec2rspec!(S)
@test isapprox(S, Sr, rtol=1e-6)

S = zeros(ComplexF64, Nw)
aspec2rspec!(S, Sa)
@test isapprox(S, Sr[1:Nw], rtol=1e-6)

# analytic spectrum -> real signal
# nothing


# ******************************************************************************
# 2D CPU
# ******************************************************************************
# real signal -> analytic signal:
E2 = copy(Er2)
plan = FFTW.plan_fft!(E2, [2])
rsig2asig!(E2, plan)
@test isapprox(E2, Ea2, rtol=1e-6)

# real signal -> analytic spectrum:
E2 = copy(Er2)
plan = FFTW.plan_fft!(E2, [2])
rsig2aspec!(E2, plan)
@test isapprox(E2, Sa2, rtol=1e-6)

# real spectrum -> analytic spectrum:
S2 = copy(Sr2)
rspec2aspec!(S2)
@test isapprox(S2, Sa2, rtol=1e-6)

# real spectrum -> analytic signal:
# nothing


# ------------------------------------------------------------------------------
# analytic signal -> real signal
E2 = copy(Ea2)
asig2rsig!(E2)
@test isapprox(E2, Er2, rtol=1e-6)

# analytic signal -> real spectrum
# nothing

# analytic spectrum -> real spectrum
# S2 = copy(Sa2)
# aspec2rspec!(S2)
# @test isapprox(S2, Sr2, rtol=1e-6)

# S2 = zeros(ComplexF64, (Nr, Nw))
# aspec2rspec!(S2, Sa2)
# @test isapprox(S2, Sr2[1:end, 1:Nw], rtol=1e-6)

# analytic spectrum -> real signal
# nothing


# ******************************************************************************
# 2D GPU
# ******************************************************************************
# real signal -> analytic signal:
E2gpu = copy(Er2gpu)
plan = FFTW.plan_fft!(E2gpu, [2])
rsig2asig!(E2gpu, plan)
@test isapprox(collect(E2gpu), Ea2, rtol=1e-6)

# real signal -> analytic spectrum:
E2gpu = copy(Er2gpu)
plan = FFTW.plan_fft!(E2gpu, [2])
rsig2aspec!(E2gpu, plan)
@test isapprox(collect(E2gpu), Sa2, rtol=1e-6)

# real spectrum -> analytic spectrum:
S2gpu = copy(Sr2)
rspec2aspec!(S2gpu)
@test isapprox(collect(S2gpu), Sa2, rtol=1e-6)

# real spectrum -> analytic signal:
# nothing

# ------------------------------------------------------------------------------
# analytic signal -> real signal
E2gpu = copy(Ea2gpu)
asig2rsig!(E2gpu)
@test isapprox(collect(E2gpu), Er2, rtol=1e-6)

# analytic signal -> real spectrum
# nothing

# analytic spectrum -> real spectrum
# S2 = CUDA.CuArray(Sa2)
# aspec2rspec!(S2)
# @test isapprox(collect(S2), Sr2, rtol=1e-6)

S2gpu = CUDA.zeros(ComplexF64, (Nr, Nw))
aspec2rspec!(S2gpu, Sa2gpu)
@test isapprox(collect(S2gpu), Sr2[1:end, 1:Nw], rtol=1e-6)

# analytic spectrum -> real signal
# nothing
