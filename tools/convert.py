"""Convert the 15 ECHO .pt checkpoints to one float32 binary + JSON index (no torch needed)."""
import json, pickle, sys, zipfile, collections
from pathlib import Path
import numpy as np

class Storage:
    def __init__(self, key, dtype): self.key, self.dtype = key, dtype
DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "HalfStorage": np.float16}

def load(path):
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]
    def rebuild(storage, offset, size, stride, *a):
        buf = np.frombuffer(z.read(f"{prefix}/data/{storage.key}"), dtype=storage.dtype)
        n = int(np.prod(size)) if size else 1
        if not size: return buf[offset].copy()
        return np.lib.stride_tricks.as_strided(buf[offset:], shape=size, strides=[s * buf.itemsize for s in stride]).copy()
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2": return rebuild
            if name == "_rebuild_parameter": return lambda data, *a: data
            if name in DT: return name
            if mod == "collections" and name == "OrderedDict": return collections.OrderedDict
            if mod.startswith("torch") or mod.startswith("numpy"):
                return lambda *a, **k: None
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            _, typ, key, loc, numel = pid
            return Storage(key, DT[typ] if isinstance(typ, str) else np.float32)
    return U(z.open(f"{prefix}/data.pkl")).load()

TH = {("charliehebdo",0):.56,("charliehebdo",1):.62,("charliehebdo",2):.55,("ferguson",0):.51,("ferguson",1):.53,("ferguson",2):.48,
      ("germanwings-crash",0):.66,("germanwings-crash",1):.50,("germanwings-crash",2):.55,("ottawashooting",0):.53,("ottawashooting",1):.39,
      ("ottawashooting",2):.67,("sydneysiege",0):.47,("sydneysiege",1):.60,("sydneysiege",2):.60}
KEEP = lambda k: not k.startswith("size_head") and k != "contagion_bias"
src, out = Path(sys.argv[1]), Path(sys.argv[2])
blobs, index, off = [], {"models": []}, 0
for (ev, sd), th in TH.items():
    sdict = load(src / f"main_ECHO_full_{ev}_seed{sd}.pt")["state_dict"]
    tensors = {}
    for k, v in sdict.items():
        if not KEEP(k): continue
        a = np.asarray(v, dtype=np.float32)
        tensors[k] = {"offset": off, "shape": list(a.shape)}
        blobs.append(a.ravel()); off += a.size
    index["models"].append({"name": f"{ev}_seed{sd}", "threshold": th, "tensors": tensors})
np.concatenate(blobs).astype("<f4").tofile(out / "echo_weights.bin")
(out / "echo_weights.json").write_text(json.dumps(index))
print(off, off * 4 / 1e6, "MB", list(index["models"][0]["tensors"]))
