# Running hls4ml end-to-end on NRP

This page is an end-to-end tutorial dealing with taking a Keras model all the way to hardware inference on Alveo U55C cards at the NRP cluster.

hls4ml's `VitisUnified` backend does the whole flow for a U55C: HLS synthesis, the kernel
wrapper, the `v++` link to an `.xclbin`, and a Python driver that runs it through XRT. This
page covers what NRP needs around it: storage, the tools volume, a `Job` for the long build,
and an FPGA pod for the run.

## The full workflow

| Part | What happens | Pod | FPGA? | Time |
| --- | --- | --- | --- | --- |
| [1. Setup](#part-1--setup) | PVC, build pod, hls4ml install | build | No | ~30 min |
| [2. Model to bitstream](#part-2--model-to-bitstream) | train, convert, `build(bitfile=True)` | build, then a Job | No | ~5 h |
| [3. Hardware validation](#part-3--hardware-validation) | load `.xclbin`, run, compare to Keras | FPGA pod | **Yes** | ~5 min |

Only Part 3 needs a card. Please avoid requesting a card during synthesis, because that would hold shared hardware resources idle.

**Note:**
This page assumes the basics in
[Requesting FPGAs from a Pod](/documentation/userdocs/fpgas/using-fpgas-from-pods). For an
interactive GUI environment see
[Vivado and Vitis](/documentation/userdocs/fpgas/vivado-vitis) but long synthesis runs are unsuccessful in Coder, which is why Part 2 uses a Job.


## Frequently asked questions

**Q: Will I need `xilinx.com/fpga_jtag`?** A: No. Loading an `.xclbin` uses the PCIe-side
resource only. See
[Before you request JTAG](/documentation/userdocs/fpgas/using-fpgas-from-pods#before-you-request-jtag-read-this).

**Q: `v++` crashed with a backtrace through `udev_enumerate_scan_devices`.** A: Different
bug from the documented `realloc()` crash, and the `lib/lnx64.o` workaround does not fix
it. See [section 7](#7-the-libudev-stub).

**Q: The build log shows failed timing. Is the design broken?** A: No. Vivado implements the
kernel against the platform's 300 MHz default, and Vitis then lowers the clock in the `.xclbin`
to what meets timing. `build()` prints the clock it ended up with. Please see
[section 8](#8-build-the-bitstream).

**Q: How do I create the files this page shows?** A: Heredocs. `cat > path << 'EOF'`, paste
the block, then `EOF` on its own line. The quotes around `'EOF'` stop bash expanding `$`
inside the content, which matters for the Python files.

## Versions

Verified end-to-end run as of October 2026:

| Component | Version |
| --- | --- |
| hls4ml | `fastmachinelearning/hls4ml` main, unpatched (Alveo support from [PR #1573](https://github.com/fastmachinelearning/hls4ml/pull/1573)) |
| Vitis / Vivado | 2024.2, from the `xilinx-tools` PVC |
| XRT | 2.19.194 (host and Coder image) |
| Platform | `xilinx_u55c_gen3x16_xdma_3_202210_1` |
| TensorFlow / Keras | tensorflow-cpu 2.19.1 / Keras 3.15.1 |
| Python | 3.12 |

---

# Part 1 — Setup

## 1. Prerequisites

An NRP account, a namespace, and `kubectl` — see
[Getting access](/documentation/userdocs/start/getting-started).

**The `xilinx-tools` PVC.** Vitis and Vivado exist on a shared read-only volume that the
Coder FPGA template mounts at `/tools/Xilinx`. To use it from your own pods, ask the
cluster admins on [Nautilus Support](https://nrp.ai/contact/) for a PVC pointing at that
volume in your namespace, then check with `kubectl get pvc -n YOUR-NAMESPACE`.

**`ErrImagePull` with unauthorized or denied** The registry normally allows anonymous pulls from inside the cluster. If you hit an auth error, ask on [Nautilus Support](https://nrp.ai/contact/) for an imagePullSecrets entry in your namespace. Licensing needs no action — the cluster FlexLM server at `2100@xilinxd.xilinx-dev` is reachable from any pod, and every manifest here sets `XILINXD_LICENSE_FILE`.

## 2. Create a work PVC

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hls4ml-work
  namespace: YOUR-NAMESPACE
spec:
  storageClassName: rook-cephfs
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 200Gi
```

`rook-cephfs` is RWX, so the build pod, the link Job, and the FPGA pod can all share it.
See [Ceph FS / RBD](/documentation/userdocs/storage/ceph) for the other classes.

CephFS ignores `fsGroup`, so a fresh PVC is root-owned and the image's `coder` user
(uid 1000) cannot write to it. Fix it from a throwaway pod:

```bash
kubectl run fixperms --restart=Never --image=busybox -n YOUR-NAMESPACE \
  --overrides='{"spec":{"containers":[{"name":"f","image":"busybox","command":["sh","-c","chown 1000:1000 /work && chmod 755 /work"],"securityContext":{"runAsUser":0},"volumeMounts":[{"name":"w","mountPath":"/work"}]}],"volumes":[{"name":"w","persistentVolumeClaim":{"claimName":"hls4ml-work"}}],"restartPolicy":"Never"}}'

kubectl logs fixperms -n YOUR-NAMESPACE
kubectl delete pod fixperms -n YOUR-NAMESPACE
```

**Caution:**
Do this **once, on a fresh PVC**. Do not use `chown -R`, because recursive chown over a
volume that already holds Vivado build trees walks hundreds of thousands of small files
and can take over ten minutes.

## 3. Start a build pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hls4ml-shell
  namespace: YOUR-NAMESPACE
spec:
  restartPolicy: Never
  nodeSelector:
    topology.kubernetes.io/region: us-west
  containers:
  - name: shell
    image: gitlab-registry.nrp-nautilus.io/nrp/coder-images/vivado-vitis
    command: ["sleep", "14400"]
    env:
    - { name: XILINXD_LICENSE_FILE, value: "2100@xilinxd.xilinx-dev" }
    resources:
      limits:   { memory: 32Gi, cpu: "8" }
      requests: { memory: 32Gi, cpu: "8" }
    volumeMounts:
    - { name: work,  mountPath: /work }
    - { name: tools, mountPath: /tools/Xilinx, readOnly: true }
  volumes:
  - name: work
    persistentVolumeClaim: { claimName: hls4ml-work }
  - name: tools
    persistentVolumeClaim: { claimName: xilinx-tools, readOnly: true }
```

```bash
kubectl apply -f build-pod.yaml
kubectl exec -it hls4ml-shell -n YOUR-NAMESPACE -- bash
```

**Caution:**
Pin `topology.kubernetes.io/region: us-west`. The vivado-vitis image is large enough that
a cold pull on a node that has never cached it can exceed the kubelet's pull deadline,
failing with `ImagePullBackOff` and a `context canceled` extraction error. The FPGAs and
the tools volume are both US-West anyway.

## 4. Install hls4ml

Put the virtualenv in the **container filesystem**, not on the PVC. A venv is tens of
thousands of small files and creating it on CephFS takes many minutes. The build Job in
section 8 creates its own.

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip

git clone https://github.com/fastmachinelearning/hls4ml.git /work/hls4ml
cd /work/hls4ml && git rev-parse --short HEAD    # record this

pip install -e /work/hls4ml "tensorflow-cpu<2.20" scikit-learn pandas
```

Install everything in one `pip` command. Splitting it makes pip install numpy 2.x for
hls4ml and then downgrade it for TensorFlow, resulting in a full uninstall and reinstall.
`pandas` is needed by `fetch_openml` in the next section and is not pulled in by
scikit-learn.

The hls4ml checkout stays on the PVC; `pip install -e .` only writes a link to it, so the
venv living in the container is not a problem.

**Caution:**
Do not keep your conversion script next to the hls4ml checkout. Python prepends the
*script's* directory to `sys.path`, so a repo root containing an `hls4ml/` folder shadows
the installed package as a namespace package. It fails as
`AttributeError: module 'hls4ml' has no attribute 'utils'`.

```bash
mkdir -p /work/run && cd /work/run
python3 -c "import hls4ml; print(hls4ml.__file__)"     # None means shadowed
```

**Note:**
`kubectl exec` gives you a fresh shell each time. After any reconnect, re-run
`source ~/venv/bin/activate` or rebuild the venv, if the pod itself was replaced.

---

# Part 2 — Model to bitstream

Sections 5–7 run in seconds in the build pod. Section 8 is the long build and runs as a `Job`.

## 5. Train a model

This tutorial uses the jet-tagging model from
[`hls4ml-tutorial`](https://github.com/fastmachinelearning/hls4ml-tutorial)
`1_getting_started/1a_train_keras.ipynb` — five-class classification of boosted jets from
16 high-level features, ~4,400 parameters. **Substitute your own Keras model if you have
one. Later sections only care about input and output shapes.**

```bash
cat > /work/run/train.py << 'EOF'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(0)

data = fetch_openml('hls4ml_lhc_jets_hlf')
X, y = data['data'], data['target']

le = LabelEncoder()
y = np.eye(5)[le.fit_transform(y)]
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, input_shape=(16,), name='fc1', activation='relu'))
model.add(Dense(32, name='fc2', activation='relu'))
model.add(Dense(32, name='fc3', activation='relu'))
model.add(Dense(5, name='output', activation='softmax'))
model.summary()

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_val, y_train_val, batch_size=1024, epochs=20,
          validation_split=0.25, shuffle=True)

os.makedirs('/work/model', exist_ok=True)
model.save('/work/model/jet_tagger.keras')

# 1024 test samples plus Keras predictions, for validating the hardware in Part 3
X_tb = X_test[:1024].astype(np.float32)
np.save('/work/model/X_test.npy', X_tb)
np.save('/work/model/y_keras.npy', model.predict(X_tb))
print('saved')
EOF

cd /work/run && python3 train.py
```

`fetch_openml` downloads the dataset on first call, so expect a pause before training
starts. Twenty epochs takes a couple of minutes on 8 CPUs.

## 6. The build script

```bash
cat > /work/run/build.py << 'EOF'
import sys

import hls4ml
from tensorflow import keras

model = keras.models.load_model('/work/model/jet_tagger.keras')

cfg = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='VitisUnified')
cfg['Model']['Precision'] = 'ap_fixed<16,6>'
cfg['Model']['ReuseFactor'] = 1

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=cfg,
    backend='VitisUnified',
    board='alveo-u55c',
    driver='xrt',
    clock_period=3.333,
    output_dir='/work/prj',
)
hls_model.write()
if '--write-only' in sys.argv:
    sys.exit('written')

report = hls_model.build(synth=True, bitfile=True)
print('KernelClockMHz:', report.get('KernelClockMHz'))
EOF

cd /work/run && python3 build.py --write-only
```

`board='alveo-u55c'` sets the part, the platform, the HBM bank assignment and the kernel's SLR.
`driver='xrt'` generates the PCIe driver used in Part 3. `--write-only` checks that conversion
works before you start a five-hour build.

**Note:**
For your own model: the default places the kernel in SLR2, and a model that needs more DSPs
than one SLR has (2928 on SLR2) fails placement with `Place 30-859`. Pass `kernel_slr=False`
to let Vivado spread it over several SLRs. Each model input and output also gets its own
memory port, and the U55C has 32 HBM banks.

## 7. The libudev stub

**Note:**
This is a different crash from the documented
[`realloc(): invalid pointer`](/documentation/userdocs/fpgas/using-fpgas-from-pods#vivado-crashes-mid-synthesis-with-realloc-invalid-pointer)
failure, and the `lib/lnx64.o` `LD_LIBRARY_PATH` workaround does not fix it. Both can be
applied together and neither interferes with the other.

Vivado's license manager (`libXil_lmgr11.so`) calls `udev_enumerate_scan_devices` to build
a webtalk host fingerprint. In a container the host's sysfs is mounted, so `/sys/class` is
fully populated, but there is no udev daemon and therefore no `/run/udev`. libudev walks
hundreds of devices and then crashes in `realloc`. This kills `v++ --link` on 2023.2,
2024.2 and 2025.2.

`LD_PRELOAD` does not help: `libXil_lmgr11.so` obtains libudev through `dlopen`, not
linkage — `ldd` shows no udev dependency, so lookups through the `dlopen` handle bypass
the preload. Shadowing it by soname on `LD_LIBRARY_PATH` does work.

```bash
cat > /work/udev_stub.c << 'EOF'
#include <stddef.h>
void *udev_new(void) { return (void*)1; }
void *udev_unref(void *u) { return NULL; }
void *udev_ref(void *u) { return u; }
void *udev_enumerate_new(void *u) { return (void*)1; }
void *udev_enumerate_unref(void *e) { return NULL; }
int udev_enumerate_add_match_subsystem(void *e, const char *s) { return 0; }
int udev_enumerate_add_match_property(void *e, const char *k, const char *v) { return 0; }
int udev_enumerate_scan_devices(void *e) { return 0; }
void *udev_enumerate_get_list_entry(void *e) { return NULL; }
void *udev_list_entry_get_next(void *l) { return NULL; }
const char *udev_list_entry_get_name(void *l) { return NULL; }
const char *udev_list_entry_get_value(void *l) { return NULL; }
void *udev_device_new_from_syspath(void *u, const char *p) { return NULL; }
void *udev_device_unref(void *d) { return NULL; }
const char *udev_device_get_devnode(void *d) { return NULL; }
const char *udev_device_get_sysattr_value(void *d, const char *a) { return NULL; }
const char *udev_device_get_property_value(void *d, const char *k) { return NULL; }
const char *udev_device_get_subsystem(void *d) { return NULL; }
const char *udev_device_get_sysname(void *d) { return NULL; }
void *udev_device_get_parent(void *d) { return NULL; }
EOF

mkdir -p /work/fakelib
gcc -shared -fPIC -Wl,-soname,libudev.so.1 -o /work/fakelib/libudev.so.1 /work/udev_stub.c
```

The build Job in the next section puts `/work/fakelib` first on `LD_LIBRARY_PATH`.

**Caution:**
Source the settings scripts **first**, then export `LD_LIBRARY_PATH` —
XRT's `setup.sh` overwrites it, so exporting first silently drops the stub.

## 8. Build the bitstream

**Caution:**
Run the build as a `Job`, not a bare pod. Bare pods fail with exit 137 and no
surviving events, and Coder kills long-running processes even under `screen` or `setsid`.
See [Running batch jobs](/documentation/userdocs/running/jobs).

Free the build pod first — nothing after this needs it:

```bash
exit
kubectl delete pod hls4ml-shell -n YOUR-NAMESPACE
```

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hls4ml-build
  namespace: YOUR-NAMESPACE
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        topology.kubernetes.io/region: us-west
      containers:
      - name: build
        image: gitlab-registry.nrp-nautilus.io/nrp/coder-images/vivado-vitis
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -e
            source /opt/xilinx/xrt/setup.sh
            source /tools/Xilinx/Vitis/2024.2/settings64.sh
            export LD_LIBRARY_PATH=/work/fakelib:$LD_LIBRARY_PATH
            export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
            python3 -m venv ~/venv
            source ~/venv/bin/activate
            pip install -q -e /work/hls4ml "tensorflow-cpu<2.20"
            cd /work/run
            time python build.py
        env:
        - { name: XILINXD_LICENSE_FILE, value: "2100@xilinxd.xilinx-dev" }
        resources:
          limits:   { memory: 64Gi, cpu: "8", ephemeral-storage: 50Gi }
          requests: { memory: 64Gi, cpu: "8", ephemeral-storage: 50Gi }
        volumeMounts:
        - { name: work,  mountPath: /work }
        - { name: tools, mountPath: /tools/Xilinx, readOnly: true }
      volumes:
      - name: work
        persistentVolumeClaim: { claimName: hls4ml-work }
      - name: tools
        persistentVolumeClaim: { claimName: xilinx-tools, readOnly: true }
```

`PLATFORM_REPO_PATHS` tells `v++` where the U55C platform is installed in the image.

```bash
kubectl apply -f link-job.yaml
kubectl get job hls4ml-build -n YOUR-NAMESPACE
```

About five hours for completion, depending on the node. `v++` logs sparsely during place and route, so
watch the Vivado implementation log instead:

```bash
kubectl exec -n YOUR-NAMESPACE job/hls4ml-build -- bash -c \
  'grep -E "^Phase [0-9]" /work/prj/vitis_workspace/system_link/_x/link/vivado/vpl/prj/prj.runs/impl_1/runme.log | tail -3'
```

Placement runs first and takes longest; `Pre Route Cleanup` means placement finished and
routing has begun. Peak memory was about 15 GB.

When `kubectl get job` shows `1/1`:

```bash
kubectl logs job/hls4ml-build -n YOUR-NAMESPACE --tail=15
```

Look for `Kernel clock in the .xclbin: 237 MHz (clock_period asks for 300 MHz)`. The
`.xclbin` and the driver are in `/work/prj/export/`: `myproject.xclbin` and
`axi_master_driver.py`.

**Note:**
`kubectl exec` stops working once the Job's pod reaches `Succeeded`. Check timing while
the Job is still running, or read the logs afterwards from the FPGA pod in Part 3. The
PVC will still keep everything.

```bash
kubectl delete job hls4ml-build -n YOUR-NAMESPACE
```

### Note about timing

The U55C kernel clock is scalable. Vivado always implements the kernel against the
platform's 300 MHz default, and when routing misses that, Vitis lowers the clock in the
`.xclbin` to the highest frequency that meets timing. So the link log reports failed timing
at 300 MHz, and the bitstream still runs correctly at the lower clock. `build()` prints the
clock it ended up with.

`clock_period` sets the HLS schedule. This model ships at 237 MHz with `clock_period=3.333`
and at 112 MHz with `clock_period=6.66`, so keep it short.

---

# Part 3 — Hardware validation

## 9. Run on the card

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hls4ml-fpga
  namespace: YOUR-NAMESPACE
spec:
  restartPolicy: Never
  nodeSelector:
    topology.kubernetes.io/region: us-west
  containers:
  - name: shell
    image: gitlab-registry.nrp-nautilus.io/nrp/coder-images/vivado-vitis
    command: ["sleep", "3600"]
    resources:
      limits:
        memory: 8Gi
        cpu: "2"
        amd.com/xilinx_u55c_gen3x16_xdma_base_3-0: 1
      requests:
        memory: 8Gi
        cpu: "2"
    volumeMounts:
    - { name: work, mountPath: /work }
  volumes:
  - name: work
    persistentVolumeClaim: { claimName: hls4ml-work }
```

```bash
kubectl apply -f fpga-pod.yaml
kubectl exec -it hls4ml-fpga -n YOUR-NAMESPACE -- bash
```

```bash
source /opt/xilinx/xrt/setup.sh
xbutil examine
```

One device should appear with shell `xilinx_u55c_gen3x16_xdma_base_3` and
`Device Ready: Yes`.

This is a fresh container, so the venv from Part 1 is gone. That does not matter, because
`pyxrt` ships with XRT and is already on `PYTHONPATH`. numpy is the only other
requirement:

```bash
python3 -c "import numpy" 2>/dev/null || pip install --user numpy
```

```bash
cd /work/prj/export
python3 - << 'EOF'
import numpy as np
from axi_master_driver import NeuralNetworkAccelerator

X = np.load('/work/model/X_test.npy')
k = np.load('/work/model/y_keras.npy')

accel = NeuralNetworkAccelerator('myproject.xclbin', X.shape, (len(X), k.shape[1]))
y, t, rate = accel.predict(X, profile=True)

print('argmax agreement:', (y.argmax(1) == k.argmax(1)).mean())
print('max abs diff    :', np.abs(y - k).max())
EOF
```

Run from `export/`, since the driver loads the `.xclbin` from a path relative to the working
directory. Feed the script through stdin (`python3 -`): `python3 script.py` would put the
script's folder on the import path instead of `export/`. The driver takes `float32` arrays,
and the conversion to `ap_fixed` happens on the card. The batch size is fixed when the driver
is constructed.

## 10. Compare against Keras

1024 test samples:

| Metric | Value |
| --- | --- |
| Kernel clock | 237 MHz |
| Argmax agreement with Keras | 99.41% |
| Max abs difference vs Keras float32 | 0.197 |
| Throughput, kernel execution only | ~2.9M predictions/s |

Throughput times only the kernel run. It excludes the buffer copies and the `.xclbin` load.

The residual difference from Keras is `ap_fixed<16,6>` quantization, not anything specific
to running on Nautilus. The same model converted at the same precision gives the same
figures when run anywhere.

Release the card when done:

```bash
exit
kubectl delete pod hls4ml-fpga -n YOUR-NAMESPACE
```

---

## Troubleshooting

### `AttributeError: module 'hls4ml' has no attribute 'utils'`

A directory named `hls4ml` is shadowing the installed package. Python prepends the
*script's* directory to `sys.path`, so this happens even when your working directory is
elsewhere. Move the script somewhere with no `hls4ml` folder beside it. Confirm with
`python3 -c "import hls4ml; print(hls4ml.__file__)"`. `None` means shadowed.

### `ImportError: fetch_openml requires pandas`

`pandas` is not a hard dependency of scikit-learn. Install it with `pip install pandas`.

### Placement fails with `Place 30-859`

The model does not fit in the SLR the kernel is assigned to. Pass `kernel_slr=False` in
`build.py`.

### `v++` segfaults during link, backtrace shows `udev_enumerate_scan_devices`

The stub is not on `LD_LIBRARY_PATH`. Check that `/work/fakelib` is first and that you
exported it *after* sourcing both settings scripts. Verify with
`grep udev /proc/<vivado-pid>/maps`, which should return nothing. This is not the
`realloc()` crash documented on
[Requesting FPGAs from a Pod](/documentation/userdocs/fpgas/using-fpgas-from-pods#vivado-crashes-mid-synthesis-with-realloc-invalid-pointer).

### `cannot exec into a container in a completed pod`

The build Job finished and its pod is `Succeeded`. Read its output with
`kubectl logs job/hls4ml-build`, or inspect files from any pod that mounts the same PVC.

### Pod died with exit 137 and no events

A bare pod was reaped. Use a `Job` for anything long-running.

### `ImagePullBackOff` with `context canceled` during layer extraction

The vivado-vitis image is large and the node had no cached copy. Pin
`topology.kubernetes.io/region: us-west`.

### Pod stuck in `ContainerCreating` with `MountVolume.MountDevice failed ... DeadlineExceeded`

The node could not mount the CephFS volume. This is on the storage side. Delete the pod and
create it again; if it keeps failing on several nodes, report it on
[Nautilus Support](https://nrp.ai/contact/) with the PVC name and the error.

### Permission denied writing to `/work`

CephFS ignores `fsGroup`. Run the one-off chown from section 2.

### Creating the venv takes forever

It is on CephFS. Put it in the container filesystem (`~/venv`) instead.

### `python3` cannot find hls4ml after reconnecting

`kubectl exec` gives a fresh shell. Re-run `source ~/venv/bin/activate`.

### `Cannot checkout license`

Confirm `XILINXD_LICENSE_FILE=2100@xilinxd.xilinx-dev` is set in the shell where `v++`
runs, not only in the pod spec. Test with `lmutil lmstat -c "$XILINXD_LICENSE_FILE"`. See
also
[Using Vivado/Vitis with the cluster license server](/documentation/userdocs/fpgas/using-fpgas-from-pods#using-vivadovitis-with-the-cluster-license-server).

### `source /tools/Xilinx/Vitis/2024.2/settings64.sh` — no such file

The tools PVC contents change over time. `ls /tools/Xilinx/Vitis/` and use what is
actually there; other versions have not been tested with this flow.
