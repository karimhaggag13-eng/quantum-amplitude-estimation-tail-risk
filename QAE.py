# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""
Real-data finance hardware demo on IBM Quantum hardware

Uses real BTC/ETH data, compresses the calibrated loss model to 4 outcomes,
and compares:
- Exact tail probability
- Classical Monte Carlo
- Quantum hardware estimate

Models:
- Gaussian copula
- t-copula
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, t as student_t, chi2

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, RYGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager


# ============================================================
# SETTINGS
# ============================================================
tickers = ["BTC-USD", "ETH-USD"]
start = "2018-01-01"
end = None
interval = "1d"

w = np.array([0.5, 0.5])

# Classical calibration size
N_model = 100_000
nu_tcopula = 2
eps = 1e-12

# More hardware-friendly compression
n_bins_hardware = 4
q_low = 0.001
q_high = 0.999

# Use a milder threshold region
alpha_var = 0.95

# Classical MC benchmark
MC_SAMPLES = 300
MC_REPEATS = 30

# Hardware settings
SHOTS = 4000
OPT_LEVEL = 1
N_HW_REPEATS = 5
USE_LEAST_BUSY = True
BACKEND_NAME = "ibm_kingston"


# ============================================================
# CLEAN REPORT MITIGATION COMPARISON SETTINGS
# ============================================================

# For the clean report, compare:
# 1. Raw hardware
# 2. Hardware + dynamical decoupling + measurement twirling
# 3. Hardware + ZNE

DD_SEQUENCE_TYPE = "XpXm"

TWIRL_NUM_RANDOMIZATIONS = 16
TWIRL_SHOTS_PER_RANDOMIZATION = max(1, SHOTS // TWIRL_NUM_RANDOMIZATIONS)

# Keep ZNE light unless you have a lot of IBM runtime.
REPORT_ZNE_NOISE_FACTORS = [1, 3]
# You can use [1, 3, 5] if you have enough IBM time.

ZNE_FIT_ORDER = 1

def make_folded_circuit(measured_circuit, scale_factor):
    """
    Global unitary folding for manual ZNE.

    scale_factor = 1 gives U
    scale_factor = 3 gives U U^\dagger U
    scale_factor = 5 gives U U^\dagger U U^\dagger U
    """

    if scale_factor % 2 != 1:
        raise ValueError("ZNE scale_factor must be odd: 1, 3, 5, ...")

    if scale_factor == 1:
        folded = measured_circuit.copy()
        folded.name = measured_circuit.name + "_zne1"
        return folded

    # Remove final measurements before folding
    base = measured_circuit.remove_final_measurements(inplace=False)

    folded = QuantumCircuit(
        measured_circuit.num_qubits,
        measured_circuit.num_clbits,
        name=f"{measured_circuit.name}_zne{scale_factor}"
    )

    qubits = list(range(measured_circuit.num_qubits))

    # Start with U
    folded.compose(base, qubits=qubits, inplace=True)

    # Add (U^\dagger U) blocks
    n_folds = (scale_factor - 1) // 2
    base_inv = base.inverse()

    for _ in range(n_folds):
        folded.compose(base_inv, qubits=qubits, inplace=True)
        folded.compose(base, qubits=qubits, inplace=True)

    # Measure objective qubit into classical bit 0
    objective_qubit = measured_circuit.num_qubits - 1
    folded.measure(objective_qubit, 0)

    return folded


def extrapolate_zero_noise(noise_factors, estimates, fit_order=1):
    """
    Fits p(lambda) as a polynomial in the noise scale lambda
    and evaluates the fitted curve at lambda = 0.
    """

    noise_factors = np.asarray(noise_factors, dtype=float)
    estimates = np.asarray(estimates, dtype=float)

    degree = min(fit_order, len(noise_factors) - 1)

    coeffs = np.polyfit(noise_factors, estimates, deg=degree)
    p_zero_noise = np.polyval(coeffs, 0.0)

    # Probabilities must stay in [0,1]
    p_zero_noise = float(np.clip(p_zero_noise, 0.0, 1.0))

    return p_zero_noise, coeffs


def run_zne_hardware_estimate(
    sampler,
    backend,
    circuit,
    noise_factors=[1, 3],
    n_repeats=1,
    shots=4000,
    opt_level=1,
    fit_order=1
):
    """
    Runs the circuit at different artificial noise scales,
    then extrapolates back to zero noise.
    """

    scale_estimates = []
    all_runs = {}

    for scale in noise_factors:
        folded_circuit = make_folded_circuit(circuit, scale)

        est, runs, counts = run_repeated_hardware_estimates(
            sampler,
            backend,
            folded_circuit,
            n_repeats=n_repeats,
            shots=shots,
            opt_level=opt_level
        )

        scale_estimates.append(est)
        all_runs[scale] = runs

        print(f"ZNE scale {scale}: estimate = {est:.6f}")

    zne_est, coeffs = extrapolate_zero_noise(
        noise_factors,
        scale_estimates,
        fit_order=fit_order
    )

    print("ZNE raw scale estimates:", dict(zip(noise_factors, scale_estimates)))
    print("ZNE extrapolated estimate:", zne_est)

    return zne_est, scale_estimates, all_runs, coeffs
# ============================================================
# DOWNLOAD REAL DATA
# ============================================================
try:
    import yfinance as yf
except ImportError:
    raise ImportError("Install yfinance first: pip install yfinance")

frames = []
for tkr in tickers:
    tmp = yf.download(
        tkr,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=30
    )
    if tmp.empty:
        raise ValueError(f"Download failed for {tkr}")

    col = tmp["Adj Close"] if "Adj Close" in tmp.columns else tmp["Close"]

    # make sure we append a Series with the ticker as its name
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]

    col.name = tkr
    frames.append(col)

prices = pd.concat(frames, axis=1).dropna()
rets = np.log(prices).diff().dropna()

if prices.empty or rets.empty or rets.shape[1] != 2:
    raise ValueError("No valid return data after download.")

print(f"Rows={len(rets)}, tickers={tickers}")


# ============================================================
# FIT STUDENT-t MARGINALS + DEPENDENCE
# ============================================================
marg_params = []
U_cols = []

for k in range(2):
    r = rets.iloc[:, k].to_numpy()
    dfk, lock, scalek = student_t.fit(r)
    marg_params.append((dfk, lock, scalek))

    u = student_t.cdf(r, dfk, loc=lock, scale=scalek)
    u = np.clip(u, eps, 1 - eps)
    U_cols.append(u)

U_data = np.column_stack(U_cols)

Z_scores = norm.ppf(U_data)
R_hat = np.corrcoef(Z_scores.T)
R_hat = np.clip(R_hat, -0.999, 0.999)
np.fill_diagonal(R_hat, 1.0)

print("\nEstimated dependence (Gaussian-score corr):")
print(pd.DataFrame(R_hat, index=tickers, columns=tickers).round(3))

print("\nFitted Student-t marginals (df, loc, scale):")
for tkr, (dfk, lock, scalek) in zip(tickers, marg_params):
    print(f"  {tkr:8s} df={dfk:.3f}, loc={lock:.6g}, scale={scalek:.6g}")


# ============================================================
# HELPERS: LOSS MODEL
# ============================================================
def simulate_loss_model(copula_type="gaussian", nu=2, Nsim=100_000, seed_local=123):
    rng_local = np.random.default_rng(seed_local)

    if copula_type == "gaussian":
        Z = rng_local.multivariate_normal(mean=np.zeros(2), cov=R_hat, size=Nsim)
        U = np.clip(norm.cdf(Z), eps, 1 - eps)

    elif copula_type == "t":
        Z = rng_local.multivariate_normal(mean=np.zeros(2), cov=R_hat, size=Nsim)
        W = chi2.rvs(df=nu, size=Nsim, random_state=rng_local)
        X = Z / np.sqrt(W / nu)[:, None]
        U = np.clip(student_t.cdf(X, df=nu), eps, 1 - eps)

    else:
        raise ValueError("copula_type must be 'gaussian' or 't'")

    Rsim = np.column_stack([
        student_t.ppf(U[:, j],
                      df=marg_params[j][0],
                      loc=marg_params[j][1],
                      scale=marg_params[j][2])
        for j in range(2)
    ])

    return -(Rsim @ w)


def compress_loss_model(loss_sample, n_bins=4, q_low=0.001, q_high=0.999):
    loss_min = np.quantile(loss_sample, q_low)
    loss_max = np.quantile(loss_sample, q_high)

    loss_clip = np.clip(loss_sample, loss_min, loss_max)
    hist, edges = np.histogram(
        loss_clip,
        bins=n_bins,
        range=(loss_min, loss_max),
        density=False
    )

    probs = hist.astype(float) / hist.sum()
    probs = probs / probs.sum()
    loss_vals = 0.5 * (edges[:-1] + edges[1:])
    return loss_vals, probs


def discrete_var_cvar(loss_vals, p, alpha=0.75):
    cdf = np.cumsum(p)
    idx = int(np.searchsorted(cdf, alpha, side="left"))
    idx = min(idx, len(loss_vals) - 1)
    var = loss_vals[idx]

    tail_mask = loss_vals >= var
    p_tail = p[tail_mask].sum()
    cvar = np.sum(loss_vals[tail_mask] * p[tail_mask]) / p_tail
    return var, cvar


def exact_tail_probability(losses, probabilities, thr):
    return float(np.sum(probabilities[losses >= thr]))


# ============================================================
# BUILD REAL-DATA CALIBRATED GAUSSIAN / t-COPULA MODELS
# ============================================================
loss_gauss_sample = simulate_loss_model("gaussian", Nsim=N_model, seed_local=101)
loss_t_sample = simulate_loss_model("t", nu=nu_tcopula, Nsim=N_model, seed_local=202)

loss_vals_gauss, probs_gauss = compress_loss_model(
    loss_gauss_sample, n_bins=n_bins_hardware, q_low=q_low, q_high=q_high
)
loss_vals_t, probs_t = compress_loss_model(
    loss_t_sample, n_bins=n_bins_hardware, q_low=q_low, q_high=q_high
)

VaR_gauss, CVaR_gauss = discrete_var_cvar(loss_vals_gauss, probs_gauss, alpha=alpha_var)
VaR_t, CVaR_t = discrete_var_cvar(loss_vals_t, probs_t, alpha=alpha_var)

# Milder common threshold from Gaussian 75% quantile on compressed grid
cdf_gauss = np.cumsum(probs_gauss)
idx_thr = int(np.searchsorted(cdf_gauss, alpha_var, side="left"))
idx_thr = min(idx_thr, len(loss_vals_gauss) - 1)

# use the right edge of the selected bin
loss_min = np.quantile(loss_gauss_sample, q_low)
loss_max = np.quantile(loss_gauss_sample, q_high)
edges = np.linspace(loss_min, loss_max, n_bins_hardware + 1)
threshold = edges[idx_thr + 1]

print("\nCompressed Gaussian loss grid:", np.round(loss_vals_gauss, 4))
print("Compressed t-copula loss grid:", np.round(loss_vals_t, 4))
print(f"Threshold used for tail event = {threshold:.4%}")


# ============================================================
# CLASSICAL MC HELPERS
# ============================================================
def mc_sample_losses(loss_vals, probabilities, M=300, seed_local=123):
    rng_local = np.random.default_rng(seed_local)
    idx = rng_local.choice(len(loss_vals), size=M, p=probabilities)
    return loss_vals[idx]


def mc_tail_probability(loss_vals, probabilities, thr, M=300, seed_local=123):
    sample = mc_sample_losses(loss_vals, probabilities, M=M, seed_local=seed_local)
    return np.mean(sample >= thr)


def mc_summary(loss_vals, probabilities, thr, exact_val, M=300, repeats=30):
    vals = []
    errs = []
    for rep in range(repeats):
        est = mc_tail_probability(loss_vals, probabilities, thr, M=M, seed_local=1000 + rep)
        vals.append(est)
        errs.append(abs(est - exact_val))
    return np.mean(vals), np.std(vals), np.mean(errs), np.std(errs)


# ============================================================
# QUANTUM HARDWARE HELPERS
# ============================================================
def basis_bits_little_endian(i: int, n: int):
    return list(map(int, format(i, f"0{n}b")[::-1]))


def controlled_on_basis_state_ry(qc, state_idx, controls, target, theta):
    bits = basis_bits_little_endian(state_idx, len(controls))

    for qb, bit in zip(controls, bits):
        if bit == 0:
            qc.x(qb)

    cgate = RYGate(theta).control(len(controls))
    qc.append(cgate, controls + [target])

    for qb, bit in zip(controls, bits):
        if bit == 0:
            qc.x(qb)


def build_expectation_circuit(probabilities, fvals, name="exp_circuit"):
    probabilities = np.asarray(probabilities, dtype=float)
    fvals = np.asarray(fvals, dtype=float)

    if len(probabilities) != len(fvals):
        raise ValueError("probabilities and fvals must have same length")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("probabilities must sum to 1")

    n = int(np.log2(len(probabilities)))
    if 2**n != len(probabilities):
        raise ValueError("Length must be a power of 2")

    qc = QuantumCircuit(n + 1, 1, name=name)
    controls = list(range(n))
    obj = n

    stateprep = StatePreparation(np.sqrt(probabilities))
    qc.compose(stateprep, qubits=controls, inplace=True)

    for i, fi in enumerate(fvals):
        fi = float(np.clip(fi, 0.0, 1.0))
        if fi <= 1e-15:
            continue
        theta = 2 * np.arcsin(np.sqrt(fi))
        controlled_on_basis_state_ry(qc, i, controls, obj, theta)

    qc.measure(obj, 0)
    return qc


def run_sampler_and_extract_prob1(sampler, backend, circuit, shots=12000, opt_level=1):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level)
    isa_circuit = pm.run(circuit)

    job = sampler.run([isa_circuit], shots=shots)
    print(f"Submitted {circuit.name}: {job.job_id()}")

    result = job.result()
    pub = result[0]
    counts = pub.data.c.get_counts()

    total = sum(counts.values())
    p1 = counts.get("1", 0) / total
    return p1, counts


def run_repeated_hardware_estimates(sampler, backend, circuit, n_repeats=5, shots=12000, opt_level=1):
    ests = []
    counts_list = []

    for _ in range(n_repeats):
        est, counts = run_sampler_and_extract_prob1(
            sampler, backend, circuit, shots=shots, opt_level=opt_level
        )
        ests.append(est)
        counts_list.append(counts)

    return float(np.median(ests)), ests, counts_list


# ============================================================
# EXACT VALUES
# ============================================================
gauss_tail_exact = exact_tail_probability(loss_vals_gauss, probs_gauss, threshold)
t_tail_exact = exact_tail_probability(loss_vals_t, probs_t, threshold)

print("\n==============================")
print("EXACT VALUES")
print("==============================")
print(f"Gaussian: VaR {alpha_var:.1%} = {VaR_gauss:.4%} | CVaR {alpha_var:.1%} = {CVaR_gauss:.4%}")
print(f"t-copula: VaR {alpha_var:.1%} = {VaR_t:.4%} | CVaR {alpha_var:.1%} = {CVaR_t:.4%}")
print(f"Gaussian exact tail probability = {gauss_tail_exact:.6f}")
print(f"t-copula exact tail probability = {t_tail_exact:.6f}")


# ============================================================
# CLASSICAL MC RESULTS
# ============================================================
gauss_tail_mc_mean, gauss_tail_mc_std, gauss_tail_mc_err_mean, gauss_tail_mc_err_std = mc_summary(
    loss_vals_gauss, probs_gauss, threshold, gauss_tail_exact,
    M=MC_SAMPLES, repeats=MC_REPEATS
)
t_tail_mc_mean, t_tail_mc_std, t_tail_mc_err_mean, t_tail_mc_err_std = mc_summary(
    loss_vals_t, probs_t, threshold, t_tail_exact,
    M=MC_SAMPLES, repeats=MC_REPEATS
)

print("\n==============================")
print("CLASSICAL MC RESULTS")
print("==============================")
print(f"Gaussian tail prob: exact={gauss_tail_exact:.6f}, MC={gauss_tail_mc_mean:.6f}, abs err={gauss_tail_mc_err_mean:.6f}")
print(f"t-copula tail prob: exact={t_tail_exact:.6f}, MC={t_tail_mc_mean:.6f}, abs err={t_tail_mc_err_mean:.6f}")


# ============================================================
# BUILD QUANTUM CIRCUITS
# ============================================================
tail_indicator_gauss = (loss_vals_gauss >= threshold).astype(float)
tail_indicator_t = (loss_vals_t >= threshold).astype(float)

qc_gauss_tail = build_expectation_circuit(
    probs_gauss, tail_indicator_gauss, name="gauss_tail_prob"
)
qc_t_tail = build_expectation_circuit(
    probs_t, tail_indicator_t, name="t_tail_prob"
)

def build_sampler_for_mode(
    backend,
    use_dynamical_decoupling=False,
    use_measurement_twirling=False
):
    """
    Builds a SamplerV2 object for one hardware mode.

    Clean report modes:
        - Raw HW: no DD, no measurement twirling
        - HW + DD + MT: DD enabled, measurement twirling enabled
        - HW + ZNE: no DD/MT, but manual ZNE through circuit folding
    """

    sampler = Sampler(mode=backend)

    try:
        sampler.options.default_shots = SHOTS
    except Exception:
        pass

    try:
        sampler.options.execution.init_qubits = True
    except Exception:
        pass

    # Dynamical decoupling
    if use_dynamical_decoupling:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = DD_SEQUENCE_TYPE

    # Measurement twirling
    if use_measurement_twirling:
        sampler.options.twirling.enable_measure = True
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.num_randomizations = TWIRL_NUM_RANDOMIZATIONS
        sampler.options.twirling.shots_per_randomization = TWIRL_SHOTS_PER_RANDOMIZATION

    return sampler







# ============================================================
# CONNECT TO IBM HARDWARE
# ============================================================
service = QiskitRuntimeService()

if USE_LEAST_BUSY:
    backend = service.least_busy(operational=True, simulator=False)
else:
    backend = service.backend(BACKEND_NAME)

print("\nUsing backend:", backend.name)

print("\nUsing backend:", backend.name)


# ============================================================
# RUN CLEAN REPORT MITIGATION COMPARISON
# ============================================================

MITIGATION_MODES = [
    {
        "label": "Raw HW",
        "use_zne": False,
        "use_dynamical_decoupling": False,
        "use_measurement_twirling": False,
        "zne_noise_factors": None,
    },
    {
        "label": "HW + DD + MT",
        "use_zne": False,
        "use_dynamical_decoupling": True,
        "use_measurement_twirling": True,
        "zne_noise_factors": None,
    },
    {
        "label": "HW + ZNE",
        "use_zne": True,
        "use_dynamical_decoupling": False,
        "use_measurement_twirling": False,
        "zne_noise_factors": REPORT_ZNE_NOISE_FACTORS,
    },
]

mitigation_results = {}

for mode in MITIGATION_MODES:
    label = mode["label"]

    print("\n==============================")
    print(f"RUNNING MITIGATION MODE: {label}")
    print("==============================")
    print("Use ZNE:", mode["use_zne"])
    print("Use DD:", mode["use_dynamical_decoupling"])
    print("Use measurement twirling:", mode["use_measurement_twirling"])

    sampler = build_sampler_for_mode(
        backend,
        use_dynamical_decoupling=mode["use_dynamical_decoupling"],
        use_measurement_twirling=mode["use_measurement_twirling"]
    )

    if mode["use_zne"]:
        print("\nRunning Gaussian circuit with ZNE...")

        gauss_zne_est, gauss_zne_scale_estimates, gauss_zne_runs, gauss_zne_coeffs = run_zne_hardware_estimate(
            sampler,
            backend,
            qc_gauss_tail,
            noise_factors=mode["zne_noise_factors"],
            n_repeats=N_HW_REPEATS,
            shots=SHOTS,
            opt_level=OPT_LEVEL,
            fit_order=ZNE_FIT_ORDER
        )

        print("\nRunning t-copula circuit with ZNE...")

        t_zne_est, t_zne_scale_estimates, t_zne_runs, t_zne_coeffs = run_zne_hardware_estimate(
            sampler,
            backend,
            qc_t_tail,
            noise_factors=mode["zne_noise_factors"],
            n_repeats=N_HW_REPEATS,
            shots=SHOTS,
            opt_level=OPT_LEVEL,
            fit_order=ZNE_FIT_ORDER
        )

        gauss_final_est = gauss_zne_est
        t_final_est = t_zne_est

    else:
        print("\nRunning Gaussian circuit...")

        gauss_final_est, gauss_runs, gauss_counts = run_repeated_hardware_estimates(
            sampler,
            backend,
            qc_gauss_tail,
            n_repeats=N_HW_REPEATS,
            shots=SHOTS,
            opt_level=OPT_LEVEL
        )

        print("\nRunning t-copula circuit...")

        t_final_est, t_runs, t_counts = run_repeated_hardware_estimates(
            sampler,
            backend,
            qc_t_tail,
            n_repeats=N_HW_REPEATS,
            shots=SHOTS,
            opt_level=OPT_LEVEL
        )

        gauss_zne_scale_estimates = None
        t_zne_scale_estimates = None
        gauss_zne_coeffs = None
        t_zne_coeffs = None

    mitigation_results[label] = {
        "gauss_est": float(gauss_final_est),
        "t_est": float(t_final_est),

        "gauss_abs_error": float(abs(gauss_final_est - gauss_tail_exact)),
        "t_abs_error": float(abs(t_final_est - t_tail_exact)),

        "gauss_zne_scale_estimates": gauss_zne_scale_estimates,
        "t_zne_scale_estimates": t_zne_scale_estimates,

        "gauss_zne_coeffs": gauss_zne_coeffs,
        "t_zne_coeffs": t_zne_coeffs,
    }

    print("\nResult summary:")
    print(
        f"Gaussian: estimate={gauss_final_est:.6f}, "
        f"exact={gauss_tail_exact:.6f}, "
        f"error={abs(gauss_final_est - gauss_tail_exact):.6f}"
    )
    print(
        f"t-copula: estimate={t_final_est:.6f}, "
        f"exact={t_tail_exact:.6f}, "
        f"error={abs(t_final_est - t_tail_exact):.6f}"
    )


# ============================================================
# SAVE MITIGATION COMPARISON RESULTS
# ============================================================

mitigation_df = pd.DataFrame([
    {
        "method": label,

        "gauss_exact": gauss_tail_exact,
        "gauss_est": res["gauss_est"],
        "gauss_abs_error": res["gauss_abs_error"],

        "t_exact": t_tail_exact,
        "t_est": res["t_est"],
        "t_abs_error": res["t_abs_error"],

        "alpha_var": alpha_var,
        "n_bins_hardware": n_bins_hardware,
        "shots": SHOTS,
        "n_hw_repeats": N_HW_REPEATS,
    }
    for label, res in mitigation_results.items()
])

print("\n==============================")
print("FINAL MITIGATION COMPARISON")
print("==============================")
print(mitigation_df)

mitigation_df.to_csv("clean_report_mitigation_results.csv", index=False)


# ============================================================
# CONVENIENCE VARIABLES FOR PLOTS
# ============================================================

gauss_tail_hw_raw = mitigation_results["Raw HW"]["gauss_est"]
t_tail_hw_raw = mitigation_results["Raw HW"]["t_est"]

gauss_tail_hw_ddmt = mitigation_results["HW + DD + MT"]["gauss_est"]
t_tail_hw_ddmt = mitigation_results["HW + DD + MT"]["t_est"]

gauss_tail_hw_zne = mitigation_results["HW + ZNE"]["gauss_est"]
t_tail_hw_zne = mitigation_results["HW + ZNE"]["t_est"]

gauss_tail_hw_raw_err = mitigation_results["Raw HW"]["gauss_abs_error"]
t_tail_hw_raw_err = mitigation_results["Raw HW"]["t_abs_error"]

gauss_tail_hw_ddmt_err = mitigation_results["HW + DD + MT"]["gauss_abs_error"]
t_tail_hw_ddmt_err = mitigation_results["HW + DD + MT"]["t_abs_error"]

gauss_tail_hw_zne_err = mitigation_results["HW + ZNE"]["gauss_abs_error"]
t_tail_hw_zne_err = mitigation_results["HW + ZNE"]["t_abs_error"]

gauss_zne_scale_estimates = mitigation_results["HW + ZNE"]["gauss_zne_scale_estimates"]
t_zne_scale_estimates = mitigation_results["HW + ZNE"]["t_zne_scale_estimates"]

gauss_zne_coeffs = mitigation_results["HW + ZNE"]["gauss_zne_coeffs"]
t_zne_coeffs = mitigation_results["HW + ZNE"]["t_zne_coeffs"]

# ============================================================
# PLOTS
# ============================================================

# Plot 1: Tail risk under Gaussian and t-copula
fig, ax = plt.subplots(figsize=(8, 5.2))

labels = ["Gaussian", f"t-copula\n(nu={nu_tcopula})"]
x = np.arange(len(labels))
width = 0.34

var_vals = [VaR_gauss, VaR_t]
cvar_vals = [CVaR_gauss, CVaR_t]

bars1 = ax.bar(x - width/2, var_vals, width, label=f"VaR {alpha_var:.1%}", color="#4c78a8")
bars2 = ax.bar(x + width/2, cvar_vals, width, label=f"CVaR {alpha_var:.1%}", color="#e45756")

for bars in [bars1, bars2]:
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width()/2,
            h * 1.01,
            f"{h:.2%}",
            ha="center",
            va="bottom",
            fontsize=11
        )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Portfolio loss", fontsize=12)
ax.set_title("Tail risk under Gaussian and t-copula models", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# ============================================================
# Plot 2: Exact vs MC vs hardware mitigation methods
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5.5))

labels2 = ["Gaussian", "t-copula"]
x2 = np.arange(len(labels2))

methods = [
    "Exact",
    "Classical MC",
    "Raw HW",
    "HW + DD + MT",
    "HW + ZNE",
]

gauss_vals = [
    gauss_tail_exact,
    gauss_tail_mc_mean,
    gauss_tail_hw_raw,
    gauss_tail_hw_ddmt,
    gauss_tail_hw_zne,
]

t_vals = [
    t_tail_exact,
    t_tail_mc_mean,
    t_tail_hw_raw,
    t_tail_hw_ddmt,
    t_tail_hw_zne,
]

all_vals = np.array([gauss_vals, t_vals])

colors = [
    "#7f7f7f",  # Exact
    "#4c78a8",  # Classical MC
    "#e45756",  # Raw HW
    "#72b7b2",  # HW + DD + MT
    "#f2a541",  # HW + ZNE
]

wbar = 0.15
offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * wbar

for k, method in enumerate(methods):
    bars = ax.bar(
        x2 + offsets[k],
        all_vals[:, k],
        wbar,
        label=method,
        color=colors[k]
    )
    ax.bar_label(bars, fmt="%.3f", fontsize=8)

ax.set_xticks(x2)
ax.set_xticklabels(labels2)
ax.set_ylabel("Tail probability")
ax.set_title("Exact vs MC vs hardware noise mitigation")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    "hardware_mitigation_tail_probability.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ============================================================
# Plot 3: Absolute error for mitigation methods
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5.5))

error_methods = [
    "Classical MC",
    "Raw HW",
    "HW + DD + MT",
    "HW + ZNE",
]

gauss_errs = [
    gauss_tail_mc_err_mean,
    gauss_tail_hw_raw_err,
    gauss_tail_hw_ddmt_err,
    gauss_tail_hw_zne_err,
]

t_errs = [
    t_tail_mc_err_mean,
    t_tail_hw_raw_err,
    t_tail_hw_ddmt_err,
    t_tail_hw_zne_err,
]

all_errs = np.array([gauss_errs, t_errs])

colors_err = [
    "#4c78a8",  # Classical MC
    "#e45756",  # Raw HW
    "#72b7b2",  # HW + DD + MT
    "#f2a541",  # HW + ZNE
]

wbar = 0.18
offsets = (np.arange(len(error_methods)) - (len(error_methods) - 1) / 2) * wbar

for k, method in enumerate(error_methods):
    bars = ax.bar(
        x2 + offsets[k],
        all_errs[:, k],
        wbar,
        label=method,
        color=colors_err[k]
    )
    ax.bar_label(bars, fmt="%.3f", fontsize=8)

ax.set_xticks(x2)
ax.set_xticklabels(labels2)
ax.set_ylabel("Absolute error")
ax.set_title("Tail probability absolute error under noise mitigation")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    "hardware_mitigation_absolute_error.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# ============================================================
# Plot 4: Zero-Noise Extrapolation explanation plot
# ============================================================

if gauss_zne_scale_estimates is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    models = [
        {
            "name": "Gaussian",
            "noise_factors": REPORT_ZNE_NOISE_FACTORS,
            "scale_estimates": gauss_zne_scale_estimates,
            "coeffs": gauss_zne_coeffs,
            "zne_est": gauss_tail_hw_zne,
            "exact": gauss_tail_exact,
        },
        {
            "name": "t-copula",
            "noise_factors": REPORT_ZNE_NOISE_FACTORS,
            "scale_estimates": t_zne_scale_estimates,
            "coeffs": t_zne_coeffs,
            "zne_est": t_tail_hw_zne,
            "exact": t_tail_exact,
        },
    ]

    for ax, model in zip(axes, models):
        noise_factors = np.asarray(model["noise_factors"], dtype=float)
        scale_estimates = np.asarray(model["scale_estimates"], dtype=float)
        coeffs = model["coeffs"]

        # Smooth line from zero noise to maximum artificial noise
        x_line = np.linspace(0, max(noise_factors), 100)
        y_line = np.polyval(coeffs, x_line)
        y_line = np.clip(y_line, 0.0, 1.0)

        # Plot fitted noise trend
        ax.plot(
            x_line,
            y_line,
            "--",
            color="black",
            label="Linear noise fit"
        )

        # Plot measured noisy hardware points
        ax.scatter(
            noise_factors,
            scale_estimates,
            color="#e45756",
            s=80,
            label="Hardware estimates"
        )

        # Plot ZNE extrapolated value at noise scale 0
        ax.scatter(
            [0],
            [model["zne_est"]],
            color="#72b7b2",
            s=100,
            marker="D",
            label="ZNE estimate"
        )

        # Plot exact benchmark
        ax.axhline(
            model["exact"],
            color="#7f7f7f",
            linestyle=":",
            linewidth=2,
            label="Exact benchmark"
        )

        # Add labels to points
        for x, y in zip(noise_factors, scale_estimates):
            ax.text(x, y, f"  {y:.3f}", va="center", fontsize=9)

        ax.text(
            0,
            model["zne_est"],
            f"  {model['zne_est']:.3f}",
            va="center",
            fontsize=9,
            color="#2a7f77"
        )

        ax.set_title(f"ZNE extrapolation: {model['name']}")
        ax.set_xlabel("Noise scale factor")
        ax.set_ylabel("Estimated tail probability")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
