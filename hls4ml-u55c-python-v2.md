# Running hls4ml end-to-end on NRP

This page is an end-to-end tutorial dealing with taking a Keras model all the way to hardware inference on Alveo U55C cards at the NRP cluster.

hls4ml's Vitis backend stops at HLS synthesis and IP export. Three additional pieces are
needed to reach a running accelerator: a kernel wrapper bridging hls4ml's streaming interface to AXI
memory-mapped ports, a `v++` link against the U55C platform, and a host program which drives
the card through XRT.

## The full workflow

| Part | What happens | Pod | FPGA? | Time |
| --- | --- | --- | --- | --- |
| [1. Setup](#part-1--setup) | PVC, build pod, hls4ml install | build | No | ~30 min |
| [2. Model to bitstream](#part-2--model-to-bitstream) | train, convert, wrapper, `v++ -c`, `v++ -l` | build, then a Job | No | ~4–7 h |
| [3. Hardware validation](#part-3--hardware-validation) | load `.xclbin`, run, compare to Keras | FPGA pod | **Yes** | ~5 min |

Only Part 3 needs a card. Please avoid requesting a card during synthesis, because that would hold shared hardware resources idle for hours.

**Note:**
This page assumes the basics in
[Requesting FPGAs from a Pod](/documentation/userdocs/fpgas/using-fpgas-from-pods). For an
interactive GUI environment see
[Vivado and Vitis](/documentation/userdocs/fpgas/vivado-vitis) but long synthesis runs are unsuccessful in Coder, which is why Part 2 uses a `Job`.


## Frequently asked questions

**Q: Will I need `xilinx.com/fpga_jtag`?** A: No. Loading an `.xclbin` uses the PCIe-side
resource only. See
[Before you request JTAG](/documentation/userdocs/fpgas/using-fpgas-from-pods#before-you-request-jtag-read-this).

**Q: `v++` crashed with a backtrace through `udev_enumerate_scan_devices`.** A: Different
bug from the documented `realloc()` crash, and the `lib/lnx64.o` workaround does not fix
it. See [section 8](#8-the-libudev-stub).

**Q: My design failed timing. Is it broken?** A: Probably not. Vitis downscales the kernel
clock at runtime, so a design with negative setup slack still computes correctly. The
example here fails timing on every run and produces correct results. Please see
[section 9](#9-link-to-a-bitstream).

**Q: How do I create the files this page shows?** A: Heredocs. `cat > path << 'EOF'`, paste
the block, then `EOF` on its own line. The quotes around `'EOF'` stop bash expanding `$`
inside the content, which matters for the Python and C++ files.

## Versions

Verified end-to-end run as of September 2026:

| Component | Version |
| --- | --- |
| hls4ml | `fastmachinelearning/hls4ml` main (tested at `d877b975`, 1.4.0.dev39), unpatched |
| Vitis / Vivado | 2024.2, from the `xilinx-tools` PVC |
| XRT | 2.19.194 (host and Coder image) |
| Platform | `xilinx_u55c_gen3x16_xdma_3_202210_1` |
| Part | `xcu55c-fsvh2892-2L-e` |
| TensorFlow / Keras | tensorflow-cpu 2.19.1 / Keras 3.15.1 |
| Python / numpy | 3.12 / 1.26.4 |

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
and can over ten minutes.

## 3. Start a build pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hls4ml-build
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
kubectl exec -it hls4ml-build -n YOUR-NAMESPACE -- bash
```

**Caution:**
Pin `topology.kubernetes.io/region: us-west`. The vivado-vitis image is large enough that
a cold pull on a node that has never cached it can exceed the kubelet's pull deadline,
failing with `ImagePullBackOff` and a `context canceled` extraction error. The FPGAs and
the tools volume are both US-West anyway.

## 4. Install hls4ml

Put the virtualenv in the **container filesystem**, not on the PVC. A venv is tens of
thousands of small files; creating it on CephFS takes many minutes, and nothing after
Part 2 needs it.

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip

git clone https://github.com/fastmachinelearning/hls4ml.git /work/hls4ml
cd /work/hls4ml && git rev-parse --short HEAD    # record this

pip install -e . "tensorflow-cpu<2.20" scikit-learn pandas
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
`AttributeError: module 'hls4ml' has no attribute 'utils'`, not as an ImportError.

```bash
mkdir -p /work/run && cd /work/run
python3 -c "import hls4ml; print(hls4ml.__file__)"     # None means shadowed
```

**Note:**
`kubectl exec` gives you a fresh shell each time. After any reconnect, re-run
`source ~/venv/bin/activate` or rebuild the venv, if the pod itself was replaced.

---

# Part 2 — Model to bitstream

Within Part 2, sections 5–7 run in seconds and can be re-run freely. Sections 8–9 are the
long Vitis steps.

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
np.savetxt('/work/model/tb_input_features.dat', X_test[:1024], fmt='%.8f')
np.savetxt('/work/model/tb_output_predictions.dat',
           model.predict(X_test[:1024]), fmt='%.8f')
print('saved')
EOF

cd /work/run && python3 train.py
```

`fetch_openml` downloads the dataset on first call, so expect a pause before training
starts. Twenty epochs takes a couple of minutes on 8 CPUs.

## 6. Convert to HLS

```bash
cat > /work/run/convert.py << 'EOF'
import hls4ml
from tensorflow import keras

model = keras.models.load_model('/work/model/jet_tagger.keras')

cfg = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='Vitis')
cfg['Model']['Precision']   = 'fixed<16,6>'
cfg['Model']['ReuseFactor'] = 1
cfg['Model']['Strategy']    = 'Latency'
cfg['Model']['BramFactor']  = 1000000000

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=cfg,
    backend='Vitis',
    part='xcu55c-fsvh2892-2L-e',
    clock_period=5,
    io_type='io_stream',
    output_dir='/work/prj',
)
hls_model.write()
print('written')
EOF

cd /work/run && python3 convert.py
```

`write()` is enough, so there is no need to call `build()`, because `v++ -c` runs HLS synthesis
itself in section 8.

The wrapper in the next section must match the generated types:

```bash
grep -E "input_t|result_t" /work/prj/firmware/defines.h
```

```
typedef nnet::array<ap_fixed<16,6>, 16*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 5*1> result_t;
```

## 7. The kernel wrapper

hls4ml generates `myproject(hls::stream<input_t>&, hls::stream<result_t>&)`, but a Vitis
kernel needs AXI memory-mapped ports to reach HBM. This wrapper reads a batch from an
input buffer into the stream, calls the model, and writes the output stream back out.

```bash
cat > /work/prj/kernel_wrapper.h << 'EOF'
#ifndef KERNEL_WRAPPER_H
#define KERNEL_WRAPPER_H

#include "firmware/defines.h"

#define NUM_CU 1
#define NUM_WORKER 1
#define NUM_CHANNEL 16
#define BATCHSIZE 1024

#define DATA_SIZE_IN 1
#define NNET_ARRAY_DEPTH 16
#define INSTREAMSIZE (DATA_SIZE_IN * NNET_ARRAY_DEPTH)

#define DATA_SIZE_OUT 5
#define OUTSTREAMSIZE DATA_SIZE_OUT

typedef ap_fixed<16,6> in_buffer_t;
typedef ap_fixed<16,6> out_buffer_t;

#endif
EOF
```

Three values are model-dependent and must agree with `firmware/defines.h`:
`NNET_ARRAY_DEPTH` is `input_t::size` (16), `DATA_SIZE_OUT` is `result_t::size` (5), and
`in_buffer_t`/`out_buffer_t` are the model's default precision. **For a different model,
read those three off `defines.h` and change them here.** Nothing else in the wrapper
changes.

```bash
cat > /work/prj/kernel_wrapper.cpp << 'EOF'
#include "firmware/myproject.h"
#include "kernel_wrapper.h"

static void read_input(const in_buffer_t *in, hls::stream<input_t> &input, int n) {
    for (int i = 0; i < DATA_SIZE_IN; i++) {
        #pragma HLS PIPELINE
        input_t tmp;
        for (int j = 0; j < NNET_ARRAY_DEPTH; j++) {
            #pragma HLS UNROLL
            tmp[j] = in[(n * DATA_SIZE_IN * NNET_ARRAY_DEPTH) + (i * NNET_ARRAY_DEPTH) + j];
        }
        input << tmp;
    }
}

static void write_result(out_buffer_t *out, hls::stream<result_t> &output, int n) {
    int out_idx = 0;
    for (int i = 0; i < DATA_SIZE_OUT / result_t::size; i++){
        result_t tmp = output.read();
        for (int j = 0; j < result_t::size; j++) {
            #pragma HLS UNROLL
            out[(n * DATA_SIZE_OUT) + out_idx] = tmp[j];
            out_idx++;
        }
    }
}

extern "C" {
void kernel_wrapper(const unsigned int batchsize, const in_buffer_t *in, out_buffer_t *out) {
    #pragma HLS interface mode=s_axilite port=batchsize

    hls::stream<input_t> input("input");
    hls::stream<result_t> output("output");
    #pragma HLS STREAM variable=input depth=DATA_SIZE_IN
    #pragma HLS STREAM variable=output depth=1

    for (int n = 0; n < batchsize; n++) {
        #pragma HLS DATAFLOW
        read_input(in, input, n);
        myproject(input, output);
        write_result(out, output, n);
    }
}
}
EOF
```

```bash
cat > /work/prj/accelerator_card.cfg << 'EOF'
kernel=kernel_wrapper
platform=xilinx_u55c_gen3x16_xdma_3_202210_1
save-temps=1

[advanced]
prop=kernel.kernel_wrapper.kernel_flags=-std=c++11

[hls]
pre_tcl=./hls_config.tcl
clock=150000000:kernel_wrapper

[connectivity]
nk=kernel_wrapper:1

sp=kernel_wrapper_1.in:HBM[0:15]
sp=kernel_wrapper_1.out:HBM[16:31]
EOF

cat > /work/prj/hls_config.tcl << 'EOF'
config_interface -m_axi_auto_max_ports=true
config_interface -m_axi_offset slave

set_clock_uncertainty 27%
EOF
```

## 8. The libudev stub

**Note:**
This is a **different** crash from the documented
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

Now set up the environment:

**Caution:**
Source the settings scripts **first**, then export `LD_LIBRARY_PATH` —
XRT's `setup.sh` overwrites it, so exporting first drops the stub.

```bash
source /tools/Xilinx/Vitis/2024.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
export LD_LIBRARY_PATH=/work/fakelib:$LD_LIBRARY_PATH
export XILINXD_LICENSE_FILE=2100@xilinxd.xilinx-dev

echo $LD_LIBRARY_PATH        # /work/fakelib must come first
```

Compile the kernel. This runs HLS synthesis and packages the result as an `.xo`. Run it
detached so that a lack of connection does not kill it:

```bash
cd /work/prj
mkdir -p build/xo

nohup v++ -c -t hw \
  --config accelerator_card.cfg \
  --messageDb=build/kernel_wrapper.mdb \
  --temp_dir build/xo --log_dir build/xo \
  -o build/myproject_kernel.xo \
  kernel_wrapper.cpp firmware/myproject.cpp \
  -I./ -I./firmware/ -I./firmware/weights/ -I./firmware/nnet_utils/ \
  > /work/prj/vpp_compile.log 2>&1 &
```

Ten to fifteen minutes. Check with:

```bash
grep -E "Estimated Fmax|loop constraints" /work/prj/vpp_compile.log
ls -la /work/prj/build/*.xo
```

Expect `Estimated Fmax: 205.47 MHz` and `All loop constraints were satisfied`. The
per-sample latency and resource estimates are in the HLS report:

```bash
find /work/prj/build -name kernel_wrapper_csynth.rpt | head -1 | xargs sed -n '1,80p'
```

For this model: 163 cycles per sample (1.087 µs), II 75, and 14 BRAM / 2866 DSP /
20002 FF / 134965 LUT — DSP is 95% of a single SLR, which is worth knowing before the
link.

## 9. Link to a bitstream

**Caution:**
Run the link as a `Job`, not a bare pod. Bare pods are killed with exit 137 and no
surviving events, and Coder is unsuccessful with long-running processes even under `screen` or `setsid`.
See [Running batch jobs](/documentation/userdocs/running/jobs).

Free the build pod first — nothing after this needs it:

```bash
exit
kubectl delete pod hls4ml-build -n YOUR-NAMESPACE
```

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hls4ml-link
  namespace: YOUR-NAMESPACE
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        topology.kubernetes.io/region: us-west
      containers:
      - name: link
        image: gitlab-registry.nrp-nautilus.io/nrp/coder-images/vivado-vitis
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -x
            source /tools/Xilinx/Vitis/2024.2/settings64.sh
            source /opt/xilinx/xrt/setup.sh
            export LD_LIBRARY_PATH=/work/fakelib:$LD_LIBRARY_PATH
            export XILINXD_LICENSE_FILE=2100@xilinxd.xilinx-dev
            cd /work/prj
            mkdir -p build/xclbin
            time v++ -l -t hw \
              --config accelerator_card.cfg \
              --messageDb=build/kernel_wrapper.mdb \
              --temp_dir build/xclbin --log_dir build/xclbin \
              -o build/kernel_wrapper.xclbin \
              build/myproject_kernel.xo
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

```bash
kubectl apply -f link-job.yaml
kubectl get job hls4ml-link -n YOUR-NAMESPACE
```

Four to six hours, depending on the node. `v++` logs sparsely during place and route, so
watch the Vivado implementation log instead:

```bash
kubectl exec -n YOUR-NAMESPACE job/hls4ml-link -- bash -c \
  'grep -E "^Phase [0-9]" /work/prj/build/xclbin/link/vivado/vpl/prj/prj.runs/impl_1/runme.log | tail -3'
```

Placement runs first and takes longest; `Pre Route Cleanup` means placement finished and
routing has begun. Peak memory was about 15 GB.

When `kubectl get job` shows `1/1`:

```bash
kubectl logs job/hls4ml-link -n YOUR-NAMESPACE --tail=15
```

Look for `Created build/kernel_wrapper.xclbin`.

**Note:**
`kubectl exec` stops working once the Job's pod reaches `Succeeded`. Check timing while
the Job is still running, or read the logs afterwards from the FPGA pod in Part 3 — the
PVC keeps everything.

```bash
kubectl delete job hls4ml-link -n YOUR-NAMESPACE
```

### Note about timing

**This design does not meet timing**, on any run we have done. Two independent builds gave
WNS −4.5 to −4.8 ns against a 6.67 ns target, with TNS in the tens of thousands. Hold
timing is met (WHS ≈ +0.009). The failing path is in one of the dense layers'
multiply-accumulate chains, but *which* layer moves between builds, since it depends on
the trained weights and the placement seed.

That is not a broken design. Vitis handles setup violations by downscaling the kernel
clock at runtime, so the bitstream builds and computes correctly — measurements in Part 3
imply an effective clock near 100 MHz against the 150 MHz requested. To close timing,
raise `ReuseFactor`, lower the requested clock, or reduce precision.

---

# Part 3 — Hardware validation

## 10. Run on the card

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

This is a fresh container, so the venv from Part 1 is gone. That does not matter —
`pyxrt` ships with XRT and is already on `PYTHONPATH`, and numpy is the only other
requirement:

```bash
python3 -c "import numpy" 2>/dev/null || pip install --user numpy
```

```bash
cat > /work/prj/host.py << 'EOF'
#!/usr/bin/env python3

import sys
import time

import numpy as np
import pyxrt

# Must match firmware/defines.h
N_IN = 16          # input_t::size
N_OUT = 5          # result_t::size
FRAC = 16 - 6      # ap_fixed<16,6> -> 10 fractional bits
SCALE = 1 << FRAC
DTYPE = np.int16


def to_fixed(x):
    # ap_fixed defaults to AP_TRN, so this must truncate toward -inf.
    return np.clip(np.floor(x.astype(np.float64) * SCALE), -32768, 32767).astype(DTYPE)


def from_fixed(v):
    return v.astype(np.float64) / SCALE


def as_array(buf, n, dtype):
    """bo.map() returns a memoryview on some XRT builds, a numpy array on others."""
    arr = np.asarray(buf)
    if arr.dtype != dtype:
        arr = arr.view(dtype)
    return arr[:n]


def main(argv):
    if len(argv) < 3:
        print(f"usage: {argv[0]} <xclbin> <input.dat> [output.dat] [index_or_bdf]")
        return 1

    xclbin_path, input_path = argv[1], argv[2]
    output_path = argv[3] if len(argv) > 3 else "hw_results.dat"
    device_arg = argv[4] if len(argv) > 4 else "0"

    features = np.loadtxt(input_path, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, N_IN)
    n_samples, n_cols = features.shape
    if n_cols != N_IN:
        raise SystemExit(f"input has {n_cols} columns, expected {N_IN}")
    print(f"samples: {n_samples}")

    device = pyxrt.device(device_arg if ":" in device_arg else int(device_arg))
    uuid = device.load_xclbin(pyxrt.xclbin(xclbin_path))
    krnl = pyxrt.kernel(device, uuid, "kernel_wrapper")
    print(f"device: {device.get_info(pyxrt.xrt_info_device.name)}")

    # kernel_wrapper(batchsize, in, out) -> args 0, 1, 2. group_id() places each buffer
    # in the HBM bank the .cfg 'sp=' lines assigned.
    itemsize = np.dtype(DTYPE).itemsize
    bo_in = pyxrt.bo(device, n_samples * N_IN * itemsize, pyxrt.bo.normal, krnl.group_id(1))
    bo_out = pyxrt.bo(device, n_samples * N_OUT * itemsize, pyxrt.bo.normal, krnl.group_id(2))

    in_view = as_array(bo_in.map(), n_samples * N_IN, DTYPE)
    in_view[:] = to_fixed(features).reshape(-1)
    bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    t0 = time.perf_counter()
    run = krnl(np.uint32(n_samples), bo_in, bo_out)
    run.wait()
    elapsed = time.perf_counter() - t0

    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    out_view = as_array(bo_out.map(), n_samples * N_OUT, DTYPE)
    results = from_fixed(np.array(out_view)).reshape(n_samples, N_OUT)

    np.savetxt(output_path, results, fmt="%.8f")
    print(f"wrote {n_samples} results to {output_path}")
    print(f"kernel time: {elapsed:.6f} s")
    print(f"throughput : {n_samples / elapsed:.0f} pred/s   (kernel only)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
EOF

cd /work/prj
python3 host.py build/kernel_wrapper.xclbin \
                /work/model/tb_input_features.dat \
                hw_results.dat
```

**Caution:**
The float-to-fixed conversion must **truncate** rather than round. `ap_fixed` defaults to
`AP_TRN`, so assigning a float in the HLS testbench truncates toward −∞. A host that
rounds disagrees with C-simulation on roughly 20% of rows — by up to 0.13 on this model —
without ever changing an argmax, so it survives a spot check of a few rows.

## 11. Compare against Keras

```bash
python3 -c "
import numpy as np
h = np.loadtxt('/work/prj/hw_results.dat')
k = np.loadtxt('/work/model/tb_output_predictions.dat')
print('max abs diff vs Keras:', np.abs(h-k).max())
print('argmax agreement     :', (h.argmax(1)==k.argmax(1)).mean())
"
```

Two independent builds, 1024 test samples each:

| Metric | Value |
| --- | --- |
| Max abs difference vs Keras float32 | 0.197 – 0.203 |
| Argmax agreement with Keras | 99.2 – 99.4% |
| Throughput, kernel execution only | 1.17M – 1.29M predictions/s |

Throughput times only `run.wait()`; it excludes host buffer sync and xclbin load. The C++
host reports roughly 20% higher on an identical bitstream with bit-identical outputs — the
gap is Python interpreter overhead around the call, not hardware.

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

### `v++` segfaults during link, backtrace shows `udev_enumerate_scan_devices`

The stub is not on `LD_LIBRARY_PATH`. Check that `/work/fakelib` is first and that you
exported it *after* sourcing both settings scripts. Verify with
`grep udev /proc/<vivado-pid>/maps`, which should return nothing. This is not the
`realloc()` crash documented on
[Requesting FPGAs from a Pod](/documentation/userdocs/fpgas/using-fpgas-from-pods#vivado-crashes-mid-synthesis-with-realloc-invalid-pointer).

### `cannot exec into a container in a completed pod`

The link Job finished and its pod is `Succeeded`. Read its output with
`kubectl logs job/hls4ml-link`, or inspect files from any pod that mounts the same PVC.

### Pod died with exit 137 and no events

A bare pod was reaped. Use a `Job` for anything long-running.

### `ImagePullBackOff` with `context canceled` during layer extraction

The vivado-vitis image is large and the node had no cached copy. Pin
`topology.kubernetes.io/region: us-west`.

### The pod sits in `Init:0/1` for ten minutes

The `chown` init container is walking every file on the PVC. Only chown a fresh volume,
and without `-R`.

### Permission denied writing to `/work`

CephFS ignores `fsGroup`. Run the one-off chown from section 2.

### Creating the venv takes forever

It is on CephFS. Put it in the container filesystem (`~/venv`) instead. Nothing after
Part 2 needs it.

### `python3` cannot find hls4ml after reconnecting

`kubectl exec` gives a fresh shell. Re-run `source ~/venv/bin/activate`.

### `Cannot checkout license`

Confirm `XILINXD_LICENSE_FILE=2100@xilinxd.xilinx-dev` is set in the shell where `v++`
runs, not only in the pod spec. Test with `lmutil lmstat -c "$XILINXD_LICENSE_FILE"`. See
also
[Using Vivado/Vitis with the cluster license server](/documentation/userdocs/fpgas/using-fpgas-from-pods#using-vivadovitis-with-the-cluster-license-server).

### Results disagree with C-simulation on many rows, but argmax always matches

The host is rounding rather than truncating in the float-to-fixed conversion.

### `source /tools/Xilinx/Vitis/2024.2/settings64.sh` — no such file

The tools PVC contents change over time. `ls /tools/Xilinx/Vitis/` and use what is
actually there; other versions have not been tested with this flow.
