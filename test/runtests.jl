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


Ergpu = CUDA.CuArray(Er)
Eagpu = CUDA.CuArray(Ea)

Srgpu = FFTW.ifft(Ergpu)
Sagpu = FFTW.ifft(Eagpu)


Er2gpu = CUDA.CuArray(Er2)
Ea2gpu = CUDA.CuArray(Ea2)

Sr2gpu = FFTW.ifft(Er2gpu, [2])
Sa2gpu = FFTW.ifft(Ea2gpu, [2])


# ******************************************************************************
# 1D CPU
# ******************************************************************************
# real signal -> analytic signal:
E = copy(Er)
rsig2asig!(E)
@test isapprox(E, Ea, rtol=1e-6)

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
S = copy(Sr)
plan = FFTW.plan_fft!(S)
rspec2asig!(S, plan)
@test isapprox(S, Ea, rtol=1e-6)


# ------------------------------------------------------------------------------
# analytic signal -> real signal
E = copy(Ea)
asig2rsig!(E)
@test isapprox(E, Er, rtol=1e-6)


# analytic signal -> real spectrum
E = copy(Ea)
plan = FFTW.plan_fft!(E)
asig2rspec!(E, plan)
@test isapprox(E, Sr, rtol=1e-6)

S = zeros(ComplexF64, Nw)
E = copy(Ea)
plan = FFTW.plan_fft!(E)
asig2rspec!(S, E, plan)
@test isapprox(S, Sr[1:Nw], rtol=1e-6)


# analytic spectrum -> real spectrum
S = copy(Sa)
aspec2rspec!(S)
@test isapprox(S, Sr, rtol=1e-6)

S = zeros(ComplexF64, Nw)
aspec2rspec!(S, Sa)
@test isapprox(S, Sr[1:Nw], rtol=1e-6)


# analytic spectrum -> real signal
S = copy(Sa)
plan = FFTW.plan_fft!(S)
aspec2rsig!(S, plan)
@test isapprox(S, Er, rtol=1e-6)


# ******************************************************************************
# 2D CPU
# ******************************************************************************
# real signal -> analytic signal:
E2 = copy(Er2)
rsig2asig!(E2)
@test isapprox(E2, Ea2, rtol=1e-6)

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
S2 = copy(Sr2)
plan = FFTW.plan_fft!(S2, [2])
rspec2asig!(S2, plan)
@test isapprox(S2, Ea2, rtol=1e-6)


# ------------------------------------------------------------------------------
# analytic signal -> real signal
E2 = copy(Ea2)
asig2rsig!(E2)
@test isapprox(E2, Er2, rtol=1e-6)


# analytic signal -> real spectrum
E2 = copy(Ea2)
plan = FFTW.plan_fft!(E2, [2])
asig2rspec!(E2, plan)
@test isapprox(E2, Sr2, rtol=1e-6)

S2 = zeros(ComplexF64, (Nr, Nw))
E2 = copy(Ea2)
plan = FFTW.plan_fft!(E2, [2])
asig2rspec!(S2, E2, plan)
@test isapprox(S2, Sr2[1:end, 1:Nw], rtol=1e-6)


# analytic spectrum -> real spectrum
S2 = copy(Sa2)
aspec2rspec!(S2)
@test isapprox(S2, Sr2, rtol=1e-6)

S2 = zeros(ComplexF64, (Nr, Nw))
aspec2rspec!(S2, Sa2)
@test isapprox(S2, Sr2[1:end, 1:Nw], rtol=1e-6)


# analytic spectrum -> real signal
S2 = copy(Sa2)
plan = FFTW.plan_fft!(S2, [2])
aspec2rsig!(S2, plan)
@test isapprox(S2, Er2, rtol=1e-6)


# ******************************************************************************
# 1D GPU
# ******************************************************************************
# real signal -> analytic signal:
Egpu = copy(Ergpu)
rsig2asig!(Egpu)
@test isapprox(Egpu, Eagpu, rtol=1e-6)

Egpu = copy(Ergpu)
plan = FFTW.plan_fft!(Egpu)
rsig2asig!(Egpu, plan)
@test isapprox(Egpu, Eagpu, rtol=1e-6)

# real signal -> analytic spectrum:
Egpu = copy(Ergpu)
plan = FFTW.plan_fft!(Egpu)
rsig2aspec!(Egpu, plan)
@test isapprox(Egpu, Sagpu, rtol=1e-6)

# real spectrum -> analytic spectrum:
Sgpu = copy(Srgpu)
rspec2aspec!(Sgpu)
@test isapprox(Sgpu, Sagpu, rtol=1e-6)


# real spectrum -> analytic signal:
Sgpu = copy(Srgpu)
plan = FFTW.plan_fft!(Sgpu)
rspec2asig!(Sgpu, plan)
@test isapprox(Sgpu, Eagpu, rtol=1e-6)


# ------------------------------------------------------------------------------
# analytic signal -> real signal
Egpu = copy(Eagpu)
asig2rsig!(Egpu)
@test isapprox(Egpu, Ergpu, rtol=1e-6)


# analytic signal -> real spectrum
# Egpu = copy(Eagpu)
# plan = FFTW.plan_fft!(Egpu)
# asig2rspec!(Egpu, plan)
# @test isapprox(Egpu, Srgpu, rtol=1e-6)

Sgpu = CUDA.zeros(ComplexF64, Nw)
Egpu = copy(Eagpu)
plan = FFTW.plan_fft!(Egpu)
asig2rspec!(Sgpu, Egpu, plan)
@test isapprox(collect(Sgpu), Sr[1:Nw], rtol=1e-6)


# analytic spectrum -> real spectrum
# Sgpu = copy(Sagpu)
# aspec2rspec!(Sgpu)
# @test isapprox(Sgpu, Srgpu, rtol=1e-6)

Sgpu = CUDA.zeros(ComplexF64, Nw)
aspec2rspec!(Sgpu, Sagpu)
@test isapprox(collect(Sgpu), Sr[1:Nw], rtol=1e-6)


# analytic spectrum -> real signal
# Sgpu = copy(Sagpu)
# plan = FFTW.plan_fft!(Sgpu)
# aspec2rsig!(Sgpu, plan)
# @test isapprox(Sgpu, Ergpu, rtol=1e-6)


# ******************************************************************************
# 2D GPU
# ******************************************************************************
# real signal -> analytic signal:
E2gpu = copy(Er2gpu)
rsig2asig!(E2gpu)
@test isapprox(E2gpu, Ea2gpu, rtol=1e-6)

E2gpu = copy(Er2gpu)
plan = FFTW.plan_fft!(E2gpu, [2])
rsig2asig!(E2gpu, plan)
@test isapprox(E2gpu, Ea2gpu, rtol=1e-6)


# real signal -> analytic spectrum:
E2gpu = copy(Er2gpu)
plan = FFTW.plan_fft!(E2gpu, [2])
rsig2aspec!(E2gpu, plan)
@test isapprox(E2gpu, Sa2gpu, rtol=1e-6)


# real spectrum -> analytic spectrum:
S2gpu = copy(Sr2gpu)
rspec2aspec!(S2gpu)
@test isapprox(S2gpu, Sa2gpu, rtol=1e-6)


# real spectrum -> analytic signal:
S2gpu = copy(Sr2gpu)
plan = FFTW.plan_fft!(S2gpu, [2])
rspec2asig!(S2gpu, plan)
@test isapprox(S2gpu, Ea2gpu, rtol=1e-6)


# ------------------------------------------------------------------------------
# analytic signal -> real signal
E2gpu = copy(Ea2gpu)
asig2rsig!(E2gpu)
@test isapprox(E2gpu, Er2gpu, rtol=1e-6)


# analytic signal -> real spectrum
# E2gpu = copy(Ea2gpu)
# plan = FFTW.plan_fft!(E2gpu, [2])
# asig2rspec!(E2gpu, plan)
# @test isapprox(E2gpu, Sr2gpu, rtol=1e-6)

S2gpu = CUDA.zeros(ComplexF64, (Nr, Nw))
E2gpu = copy(Ea2gpu)
plan = FFTW.plan_fft!(E2gpu, [2])
asig2rspec!(S2gpu, E2gpu, plan)
@test isapprox(collect(S2gpu), Sr2[1:end, 1:Nw], rtol=1e-6)


# analytic spectrum -> real spectrum
# S2 = CUDA.CuArray(Sa2)
# aspec2rspec!(S2)
# @test isapprox(collect(S2), Sr2, rtol=1e-6)

S2gpu = CUDA.zeros(ComplexF64, (Nr, Nw))
aspec2rspec!(S2gpu, Sa2gpu)
@test isapprox(collect(S2gpu), Sr2[1:end, 1:Nw], rtol=1e-6)


# analytic spectrum -> real signal
# nothing
