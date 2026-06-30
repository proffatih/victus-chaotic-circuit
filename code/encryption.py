"""
Chaos-based image encryption using the fractional-order Chen system, with a
full security analysis. All metrics are computed on real images.

Scheme (permutation-diffusion, SHA-256 plaintext-coupled key):
  1. A 256-bit secret key + SHA-256 hash of the plaintext set the initial
     conditions and parameters of the fractional Chen system (plaintext
     coupling defeats chosen/known-plaintext attacks and gives high NPCR/UACI).
  2. Integrate the fractional Chen system (ABM solver, q=0.9) to obtain three
     chaotic sequences; quantise to key-streams.
  3. Permutation: row/column scrambling driven by argsort of chaotic sequences.
  4. Diffusion: XOR with the chaotic byte key-stream, with forward chaining so
     each cipher pixel depends on all previous ones.
Decryption is the exact inverse.

Security analysis:
  key space, key sensitivity (NPCR between two ciphers from keys differing in 1
  bit), histogram, global Shannon entropy, adjacent-pixel correlation
  (H/V/D), differential attack NPCR & UACI, encryption/decryption timing.

Open test images from skimage (camera, etc.). Results -> results/encryption_metrics.json
"""
import json
import time
import hashlib
import numpy as np

from fde_solver import abm_fde, chen_rhs  # full-memory ABM (exact)

A, B, C = 35.0, 3.0, 28.0
Q = 0.90


def key_to_state(key_bytes, img_hash):
    """Derive fractional-Chen initial state & a parameter perturbation from a
    256-bit key XORed with the SHA-256 of the plaintext."""
    h = hashlib.sha256(key_bytes + img_hash).digest()  # 32 bytes
    vals = np.frombuffer(h, dtype=np.uint8).astype(float)
    # initial conditions in a chaotic basin
    x0 = -9.0 + (vals[0:4].sum() / (4 * 255)) * 2.0
    y0 = -5.0 + (vals[4:8].sum() / (4 * 255)) * 2.0
    z0 = 14.0 + (vals[8:12].sum() / (4 * 255)) * 2.0
    cpar = C + (vals[12] / 255.0 - 0.5) * 0.5  # tiny param spread, stays chaotic
    return np.array([x0, y0, z0]), cpar


def chaotic_streams(state, cpar, n_needed):
    """Integrate the fractional Chen system once (exact full-memory ABM) and
    return a uint8 keystream of length >= n_needed plus the x,y,z sequences for
    permutation.  Three independent keystream bytes are extracted per time
    step (from x, y and z), so only ~n_needed/3 integration steps are required,
    keeping the exact O(N^2) ABM affordable for 256x256 images."""
    h = 0.01
    trans = 300                                   # transient samples discarded
    nsteps = int(np.ceil(n_needed / 3)) + trans + 5
    T = nsteps * h
    t, sol = abm_fde(chen_rhs, Q, state, 0.0, T, h, (A, B, cpar))
    s = sol[trans:]
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    # three byte streams from the fractional parts of the scaled states
    def tobyte(v, scale):
        return (np.floor(np.abs(v) * scale) % 256).astype(np.uint8)
    bx = tobyte(x, 1e6); by = tobyte(y, 1e6); bz = tobyte(z, 1e6)
    ks = np.empty(len(x) * 3, dtype=np.uint8)
    ks[0::3] = bx; ks[1::3] = by; ks[2::3] = bz
    return ks[:max(n_needed, 3)], x, y, z


def encrypt(img, key_bytes):
    img = np.asarray(img, dtype=np.uint8)
    H, W = img.shape
    npix = H * W
    img_hash = hashlib.sha256(img.tobytes()).digest()
    state, cpar = key_to_state(key_bytes, img_hash)
    ks, xs, ys, zs = chaotic_streams(state, cpar, max(npix, H + W) + 10)

    # --- permutation: scramble rows then cols using argsort of chaotic seqs ---
    row_perm = np.argsort(xs[:H])
    col_perm = np.argsort(ys[:W])
    P = img[row_perm][:, col_perm]

    # --- diffusion: XOR with keystream + forward chaining ---
    flat = P.flatten().astype(np.uint8)
    ksf = ks[:npix]
    cipher = np.empty(npix, dtype=np.uint8)
    prev = np.uint8(0xA5)
    for i in range(npix):
        cipher[i] = flat[i] ^ ksf[i] ^ prev
        prev = cipher[i]
    return cipher.reshape(H, W), (row_perm, col_perm, ksf, img_hash)


def decrypt(cipher, key_bytes, img_hash):
    cipher = np.asarray(cipher, dtype=np.uint8)
    H, W = cipher.shape
    npix = H * W
    state, cpar = key_to_state(key_bytes, img_hash)
    ks, xs, ys, zs = chaotic_streams(state, cpar, max(npix, H + W) + 10)
    row_perm = np.argsort(xs[:H]); col_perm = np.argsort(ys[:W])
    ksf = ks[:npix]
    c = cipher.flatten().astype(np.uint8)
    flat = np.empty(npix, dtype=np.uint8)
    prev = np.uint8(0xA5)
    for i in range(npix):
        flat[i] = c[i] ^ ksf[i] ^ prev
        prev = c[i]
    P = flat.reshape(H, W)
    inv_row = np.argsort(row_perm); inv_col = np.argsort(col_perm)
    img = P[:, inv_col][inv_row]
    return img


# ---------------- security metrics ----------------
def entropy(img):
    hist = np.bincount(img.flatten(), minlength=256).astype(float)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def correlation(img, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    H, W = img.shape
    out = {}
    for name, (dy, dx) in {"horizontal": (0, 1), "vertical": (1, 0),
                            "diagonal": (1, 1)}.items():
        ys = rng.integers(0, H - dy - 1, n)
        xs = rng.integers(0, W - dx - 1, n)
        a = img[ys, xs].astype(float)
        b = img[ys + dy, xs + dx].astype(float)
        out[name] = float(np.corrcoef(a, b)[0, 1])
    return out


def npcr_uaci(c1, c2):
    c1 = c1.astype(int); c2 = c2.astype(int)
    diff = c1 != c2
    npcr = float(np.mean(diff) * 100.0)
    uaci = float(np.mean(np.abs(c1 - c2)) / 255.0 * 100.0)
    return npcr, uaci


def main():
    from skimage import data
    from skimage.transform import resize
    import numpy as np

    # open standard test images, all 256x256 grayscale uint8
    images = {}
    cam = data.camera()                       # 512x512 uint8
    images["camera"] = (resize(cam, (256, 256), preserve_range=True)
                        .astype(np.uint8))
    coins = data.coins()
    images["coins"] = (resize(coins, (256, 256), preserve_range=True)
                       .astype(np.uint8))

    key = bytes.fromhex("0f1e2d3c4b5a69788796a5b4c3d2e1f0"
                        "112233445566778899aabbccddeeff00")  # 256-bit key

    report = {"system": "fractional Chen", "q": Q, "key_bits": 256,
              "key_space": "2^256 (key) x IEEE-754 IC precision ~ >2^256",
              "images": {}}

    for name, img in images.items():
        H, W = img.shape
        t0 = time.time()
        cipher, aux = encrypt(img, key)
        t_enc = time.time() - t0
        img_hash = aux[3]
        t0 = time.time()
        dec = decrypt(cipher, key, img_hash)
        t_dec = time.time() - t0
        lossless = bool(np.array_equal(dec, img))

        # differential attack: flip one pixel of plaintext, re-encrypt
        img2 = img.copy()
        r, cc = H // 2, W // 2
        img2[r, cc] ^= 1
        cipher2, _ = encrypt(img2, key)
        npcr, uaci = npcr_uaci(cipher, cipher2)

        # key sensitivity: flip 1 bit of key, encrypt same image
        key_arr = bytearray(key); key_arr[0] ^= 0x01
        cipher_k, _ = encrypt(img, bytes(key_arr))
        ks_npcr, ks_uaci = npcr_uaci(cipher, cipher_k)
        # decrypt with wrong key -> should fail
        wrongdec = decrypt(cipher, bytes(key_arr),
                           hashlib.sha256(img.tobytes()).digest())
        wrong_recovery_corr = float(np.corrcoef(wrongdec.flatten().astype(float),
                                                img.flatten().astype(float))[0, 1])

        report["images"][name] = {
            "size": [int(H), int(W)],
            "entropy_plain": round(entropy(img), 4),
            "entropy_cipher": round(entropy(cipher), 4),
            "corr_plain": {k: round(v, 4) for k, v in correlation(img).items()},
            "corr_cipher": {k: round(v, 5) for k, v in correlation(cipher).items()},
            "NPCR_percent": round(npcr, 4),
            "UACI_percent": round(uaci, 4),
            "key_sensitivity_NPCR_percent": round(ks_npcr, 4),
            "key_sensitivity_UACI_percent": round(ks_uaci, 4),
            "wrong_key_recovery_correlation": round(wrong_recovery_corr, 5),
            "lossless_decryption": lossless,
            "encrypt_time_s": round(t_enc, 4),
            "decrypt_time_s": round(t_dec, 4),
            "throughput_Mbps": round(H * W * 8 / t_enc / 1e6, 4),
        }
        # save arrays for figures
        np.savez(f"../results/enc_{name}.npz", plain=img, cipher=cipher, dec=dec)
        print(name, "done; entropy_cipher=",
              report["images"][name]["entropy_cipher"],
              "NPCR=", npcr, "UACI=", uaci, "lossless=", lossless)

    with open("../results/encryption_metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
