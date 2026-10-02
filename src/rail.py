"""Rail corrugation: classify a 1-second axle-box recording as Normal / Side I / Side II.

Layout, confirmed by reading a file: column 0 is "Rotating speed", then for
each car 1..8 and position 1..8, a vibration column followed by a shock
column, so 1 + 128 columns. Sampling is 10 kHz for 1 second, units m/s^2.

Odd positions (1,3,5,7) ride the Side I rail, even positions (2,4,6,8) the
Side II rail, giving 32 vibration and 32 shock channels per side.

Corrugation is a periodic wear pattern, so it shows up as energy at a
characteristic frequency rather than as a raw amplitude increase. Features are
therefore spectral band energies averaged across a side's channels, plus the
per-band contrast between the two sides, which is what actually distinguishes
"Side I is bad" from "Side II is bad" from "neither is".
"""
import numpy as np
import pandas as pd

FS = 10000                      # Hz
N_BANDS = 25                    # 200 Hz per band up to Nyquist
WHEEL_DIAMETER = 0.85           # m
TEETH = 90

CLASSES = ["Normal", "Side I", "Side II"]


def load(path):
    """(speed channel, vibration array, shock array) with side-major columns."""
    df = pd.read_csv(path, dtype=np.float32)
    a = df.to_numpy(dtype=np.float32)
    speed = a[:, 0]
    body = a[:, 1:]
    # body columns run car1pos1_vib, car1pos1_shock, car1pos2_vib, ...
    vib = body[:, 0::2]         # 64 channels, ordered (car, position)
    shock = body[:, 1::2]
    return speed, vib, shock


def side_masks(n_channels=64):
    """Boolean masks selecting Side I (odd positions) and Side II (even)."""
    pos = np.arange(n_channels) % 8 + 1      # position within each car
    return (pos % 2 == 1), (pos % 2 == 0)


def band_energy(x):
    """Mean band energies across channels. x is (samples, channels)."""
    # detrend per channel so a DC offset does not land in band 0
    x = x - x.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / FS)
    edges = np.linspace(0, FS / 2, N_BANDS + 1)
    out = np.empty(N_BANDS)
    for i in range(N_BANDS):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = spec[m].mean() if m.any() else 0.0
    return out


def time_stats(x):
    """RMS, peak, kurtosis and crest factor, averaged over channels."""
    x = x - x.mean(axis=0, keepdims=True)
    rms = np.sqrt((x ** 2).mean(axis=0))
    peak = np.abs(x).max(axis=0)
    sd = x.std(axis=0) + 1e-12
    kurt = (((x / sd) ** 4).mean(axis=0))
    crest = peak / (rms + 1e-12)
    return [rms.mean(), peak.mean(), kurt.mean(), crest.mean()]


def train_speed(speed):
    """m/s from the 90-tooth wheel pulse train, via 0/1 transition count."""
    s = speed > (speed.min() + speed.max()) / 2.0
    transitions = int(np.abs(np.diff(s.astype(np.int8))).sum())
    revs = transitions / 2.0 / TEETH          # two transitions per tooth
    return revs * np.pi * WHEEL_DIAMETER      # one second of data


def features(path):
    speed, vib, shock = load(path)
    m1, m2 = side_masks(vib.shape[1])

    b1 = band_energy(vib[:, m1])
    b2 = band_energy(vib[:, m2])
    s1 = band_energy(shock[:, m1])
    s2 = band_energy(shock[:, m2])

    eps = 1e-20
    f = []
    names = []
    for tag, arr in (("vibI", b1), ("vibII", b2), ("shkI", s1), ("shkII", s2)):
        f.extend(np.log10(arr + eps))
        names.extend(["%s_b%02d" % (tag, i) for i in range(N_BANDS)])
    # per-band contrast between the two sides: the discriminative part
    f.extend(np.log10(b1 + eps) - np.log10(b2 + eps))
    names.extend(["vib_contrast_b%02d" % i for i in range(N_BANDS)])
    f.extend(np.log10(s1 + eps) - np.log10(s2 + eps))
    names.extend(["shk_contrast_b%02d" % i for i in range(N_BANDS)])

    for tag, sel in (("vibI", vib[:, m1]), ("vibII", vib[:, m2]),
                     ("shkI", shock[:, m1]), ("shkII", shock[:, m2])):
        st = time_stats(sel)
        f.extend(st)
        names.extend(["%s_%s" % (tag, k) for k in ("rms", "peak", "kurt", "crest")])

    f.append(train_speed(speed))
    names.append("speed_mps")
    return np.array(f, dtype=np.float64), names
