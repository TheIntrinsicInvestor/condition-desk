"""Rail: order-tracking features, banded by wavelength instead of frequency.

Corrugation is a periodic wear pattern on the rail, so it has a characteristic
*wavelength*. What an axle box measures is a frequency, and

    frequency = speed / wavelength

so a fixed 200 Hz band covers a completely different wavelength at every
speed. Train speed here spans 0 to 19.5 m/s, and even among the files fast
enough to be corrugated it varies by a factor of two, so the existing
frequency bands cannot be reading a consistent wavelength signature.

The fix is standard order tracking: resample each channel at constant
*distance* increments rather than constant time increments, using the 90-tooth
tacho pulse train on column 0 as the phase reference. The FFT of the resampled
signal is then a spatial spectrum in cycles per metre, speed-invariant by
construction. This also removes smearing caused by the speed changing within a
single one-second recording, which dividing band edges by a mean speed would
not.

Emits the same feature families as src/rail.py so it is a drop-in replacement:
band energies per side for vibration and shock, the per-band side contrast,
time-domain statistics, and speed.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

FS = 10000                  # Hz
WHEEL_DIAMETER = 0.85       # m
TEETH = 90
DX = 0.002                  # m between resampled points -> Nyquist 250 cyc/m
MIN_DISTANCE = 1.0          # m; below this the spatial spectrum is meaningless
N_BANDS = 25
BAND_EDGES = np.logspace(np.log10(1.0), np.log10(250.0), N_BANDS + 1)  # cycles/m

# distance advanced per tacho edge: two edges per tooth, TEETH teeth per rev
M_PER_EDGE = np.pi * WHEEL_DIAMETER / (2.0 * TEETH)


def load(path):
    a = pd.read_csv(path, dtype=np.float32).to_numpy(dtype=np.float32)
    body = a[:, 1:]
    return a[:, 0], body[:, 0::2], body[:, 1::2]


def side_masks(n_channels=64):
    pos = np.arange(n_channels) % 8 + 1
    return (pos % 2 == 1), (pos % 2 == 0)


def distance_axis(tacho):
    """Cumulative distance travelled at each sample, from the pulse train.

    Edges are located by thresholding the pulse train at its midpoint; each
    edge advances the wheel by one half-tooth of circumference. Distance
    between edges is linearly interpolated, which is the standard assumption
    and is accurate at this edge density.
    """
    lo, hi = float(tacho.min()), float(tacho.max())
    if hi - lo < 1e-6:                      # stationary: no pulses at all
        return None
    s = tacho > (lo + hi) / 2.0
    edges = np.flatnonzero(np.diff(s.astype(np.int8)) != 0) + 1
    if edges.size < 4:
        return None
    d_at_edge = np.arange(edges.size) * M_PER_EDGE
    # extend to the whole record so every sample has a distance
    return np.interp(np.arange(tacho.size), edges, d_at_edge)


def resample_to_distance(x, dist):
    """Resample columns of x onto a uniform grid in distance.

    x is (samples, channels). Decimating in space aliases unless the signal is
    smoothed first, so apply a moving average matched to the decimation ratio
    before interpolating.
    """
    total = dist[-1] - dist[0]
    n_out = int(total / DX)
    step = total / len(dist)                # mean raw metres per sample
    ratio = DX / step
    if ratio > 1.5:
        x = uniform_filter1d(x, size=int(round(ratio)), axis=0, mode="nearest")
    grid = dist[0] + np.arange(n_out) * DX
    out = np.empty((n_out, x.shape[1]), dtype=np.float64)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(grid, dist, x[:, c])
    return out


def spatial_bands(x):
    """Mean band energies over channels, banded by spatial frequency."""
    x = x - x.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
    k = np.fft.rfftfreq(x.shape[0], d=DX)            # cycles per metre
    out = np.empty(N_BANDS)
    for i in range(N_BANDS):
        m = (k >= BAND_EDGES[i]) & (k < BAND_EDGES[i + 1])
        out[i] = spec[m].mean() if m.any() else 0.0
    return out


def time_stats(x):
    x = x - x.mean(axis=0, keepdims=True)
    rms = np.sqrt((x ** 2).mean(axis=0))
    peak = np.abs(x).max(axis=0)
    sd = x.std(axis=0) + 1e-12
    kurt = ((x / sd) ** 4).mean(axis=0)
    return [rms.mean(), peak.mean(), kurt.mean(), (peak / (rms + 1e-12)).mean()]


def feature_names():
    names = []
    for tag in ("vibI", "vibII", "shkI", "shkII"):
        names += ["%s_w%02d" % (tag, i) for i in range(N_BANDS)]
    names += ["vib_wcontrast_w%02d" % i for i in range(N_BANDS)]
    names += ["shk_wcontrast_w%02d" % i for i in range(N_BANDS)]
    for tag in ("vibI", "vibII", "shkI", "shkII"):
        names += ["%s_%s" % (tag, k) for k in ("rms", "peak", "kurt", "crest")]
    names += ["speed_mps", "order_valid"]
    return names


def features(path):
    """(feature vector, names). order_valid is 0 when the file was too slow."""
    tacho, vib, shock = load(path)
    m1, m2 = side_masks(vib.shape[1])
    dist = distance_axis(tacho)

    eps = 1e-20
    n_spec = 6 * N_BANDS
    if dist is None or (dist[-1] - dist[0]) < MIN_DISTANCE:
        spec_part = np.zeros(n_spec)
        valid = 0.0
        speed = 0.0 if dist is None else (dist[-1] - dist[0])
        sel = [vib[:, m1], vib[:, m2], shock[:, m1], shock[:, m2]]
    else:
        speed = dist[-1] - dist[0]           # metres in one second = m/s
        rv = resample_to_distance(vib, dist)
        rs = resample_to_distance(shock, dist)
        b1, b2 = spatial_bands(rv[:, m1]), spatial_bands(rv[:, m2])
        s1, s2 = spatial_bands(rs[:, m1]), spatial_bands(rs[:, m2])
        spec_part = np.concatenate([
            np.log10(b1 + eps), np.log10(b2 + eps),
            np.log10(s1 + eps), np.log10(s2 + eps),
            np.log10(b1 + eps) - np.log10(b2 + eps),
            np.log10(s1 + eps) - np.log10(s2 + eps),
        ])
        valid = 1.0
        sel = [rv[:, m1], rv[:, m2], rs[:, m1], rs[:, m2]]

    f = list(spec_part)
    for x in sel:
        f.extend(time_stats(x))
    f.append(speed)
    f.append(valid)
    return np.array(f, dtype=np.float64), feature_names()
