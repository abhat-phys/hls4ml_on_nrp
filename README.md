# Running hls4ml end-to-end on NRP

This page takes a Keras model to inference on an AMD Alveo U55C FPGA card on Nautilus, using
hls4ml's `VitisUnified` backend. You convert and check the model interactively, then two Jobs do
the rest: a build Job that turns the model into an FPGA bitstream, and a short validation Job that
runs it on a card and checks the output.

| Step | Where it runs | FPGA card | Time |
| --- | --- | --- | --- |
| [1. Setup](#1-setup) | your machine, one pod | No | ~20 min, once |
| [2. Convert and check the model](#2-convert-and-check-the-model) | shell pod | No | ~10 min |
| [3. Build the bitstream](#3-build-the-bitstream) | build Job | No | ~2 h (1–3 h) |
| [4. Validate on the card](#4-validate-on-the-card) | validation Job | Yes, ~1 min | ~5 min |

Only step 4 requests a card. Please don't request one while a build runs: a card attached to an
idle pod is unavailable to everyone else.

This page assumes the basics in
[Requesting FPGAs from a Pod](/documentation/userdocs/fpgas/using-fpgas-from-pods) and
[Running batch jobs](/documentation/userdocs/running/jobs). For an interactive GUI environment see
[Vivado and Vitis](/documentation/userdocs/fpgas/vivado-vitis); long builds don't survive in Coder,
which is why step 3 is a Job.

## Terms used on this page

- **C-simulation:** hls4ml compiles the generated C++ and runs it on the CPU. `predict()` uses it,
  and it gives the exact numbers the FPGA should produce.
- **HLS synthesis (csynth):** turns the C++ into a hardware description and estimates its size
  (DSPs, LUTs) and speed. Takes minutes.
- **Link:** Vitis places and routes the design onto the card and writes the bitstream. Takes hours
  and most of the memory.
- **xclbin:** the bitstream file that is loaded onto the card.
- **SLR:** the U55C's FPGA is made of three dies (super logic regions). By default the design is
  placed in one of them, SLR2, which has 2,879 DSP blocks available.

## Frequently asked questions

**Q: Do I need `xilinx.com/fpga_jtag`?** A: No. Loading an xclbin uses only the PCIe-side resource.
See [Before you request JTAG](/documentation/userdocs/fpgas/using-fpgas-from-pods#before-you-request-jtag-read-this).

**Q: What is `with-udev-stub`?** A: A wrapper in the `vivado-vitis` image. Inside a pod, Vivado's
license manager crashes or hangs while scanning devices through libudev. `with-udev-stub` runs one
command with a no-op libudev in front of the system one. Use it on `v++` and `vitis-run` only, not on
Python or your whole shell.

**Q: The build log reports failed timing. Is the design broken?** A: No. Vivado implements the
design against the card's 300 MHz default, and Vitis lowers the clock stored in the xclbin to one
that meets timing. Step 4 prints the clock the card runs at.

**Q: How do I create the files on this page?** A: With heredocs: `cat > path << 'EOF'`, paste the
block, then `EOF` on its own line. The quotes around `'EOF'` stop bash from expanding `$` inside the
file.

## Versions

Tested end to end in November 2026:

| Component | Version |
| --- | --- |
| Image | `gitlab-registry.nrp-nautilus.io/nrp/coder-images/vivado-vitis@sha256:92345d9af99ad2aa2265578bd789a94186949196bebe9a4839600f87bab5cecb` |
| hls4ml | `fastmachinelearning/hls4ml` main, commit `8912ed2` (1.4.0.dev69) |
| quantizers | 1.2.2 (must be below 2.0, see [Troubleshooting](#troubleshooting)) |
| TensorFlow / Keras / NumPy | 2.21.0 / 3.15.1 / 1.26.4, from the image |
| Vitis / Vivado | 2024.2, from the `xilinx-tools` volume |
| XRT | 2.19.194 |
| Platform | `xilinx_u55c_gen3x16_xdma_3_202210_1` |

The manifests below pin the image by digest so a later image update can't change your results. To
reproduce our numbers exactly, replace `@main` in the `pip install` lines with `@8912ed2`.

---

## 1. Setup

### Prerequisites

You need an NRP account, a namespace and `kubectl`; see
[Getting started](/documentation/userdocs/start/getting-started).

**The `xilinx-tools` volume.** Vitis and Vivado live on a shared read-only volume. To mount it in your
own pods, ask on [Nautilus Support](https://nrp.ai/contact/) for an `xilinx-tools` PVC in your
namespace, then check that it exists:

```bash
kubectl get pvc -n YOUR-NAMESPACE
```

**Licensing** needs no action. The cluster license server at `2100@xilinxd.xilinx-dev` is reachable
from any pod, and every manifest here sets `XILINXD_LICENSE_FILE`.

### Create a work volume

This PVC holds your model, scripts and build results. It is `ReadWriteMany`, so the shell pod and
both Jobs can mount it at the same time.

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
      storage: 50Gi
```

Each build keeps about 100–300 MB on the volume. Vivado's large temporary files go to the node's
local disk and are deleted with the pod.

A new CephFS volume is owned by root, and CephFS ignores `fsGroup`, so the image's `coder` user
(uid 1000) can't write to it. Hand it over once from a throwaway pod:

```bash
kubectl run fixperms --restart=Never --image=busybox -n YOUR-NAMESPACE \
  --overrides='{"spec":{"containers":[{"name":"f","image":"busybox","command":["sh","-c","chown 1000:1000 /work && chmod 755 /work"],"securityContext":{"runAsUser":0},"volumeMounts":[{"name":"w","mountPath":"/work"}]}],"volumes":[{"name":"w","persistentVolumeClaim":{"claimName":"hls4ml-work"}}],"restartPolicy":"Never"}}'

kubectl logs fixperms -n YOUR-NAMESPACE
kubectl delete pod fixperms -n YOUR-NAMESPACE
```

**Do this once, on a new volume.** Don't use `chown -R` on a volume that already holds build trees:
walking hundreds of thousands of small files can take over ten minutes.

### Start a shell pod

You use this pod for step 2. It needs no card and no tools volume.

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
    image: gitlab-registry.nrp-nautilus.io/nrp/coder-images/vivado-vitis@sha256:92345d9af99ad2aa2265578bd789a94186949196bebe9a4839600f87bab5cecb
    command: ["sleep", "14400"]
    resources:
      limits:   { memory: 8Gi, cpu: "2" }
      requests: { memory: 8Gi, cpu: "2" }
    volumeMounts:
    - { name: work, mountPath: /work }
  volumes:
  - name: work
    persistentVolumeClaim: { claimName: hls4ml-work }
```

```bash
kubectl apply -f shell-pod.yaml
kubectl exec -it hls4ml-shell -n YOUR-NAMESPACE -- bash
```

**Keep the `us-west` node selector.** The image is large, and a first pull on a node that has never
cached it can exceed the kubelet's pull deadline (`ImagePullBackOff` with `context canceled`). The
FPGA nodes and the tools volume are in US-West anyway.

### Install hls4ml

TensorFlow and Keras already come with the image, so the virtualenv only adds hls4ml and the
training tools. Create it in the container's own filesystem, not on the volume: a venv is tens of
thousands of small files, and creating it on CephFS takes many minutes.

```bash
python3 -m venv --system-site-packages ~/venv
source ~/venv/bin/activate
pip install "hls4ml @ git+https://github.com/fastmachinelearning/hls4ml@main" "quantizers<2" scikit-learn pandas
python3 -c "import hls4ml; print(hls4ml.__version__)"
```

`kubectl exec` opens a fresh shell each time, so run `source ~/venv/bin/activate` again after you
reconnect.

---

## 2. Convert and check the model

Everything in this step runs in the shell pod and takes seconds to minutes. This is where you
choose precision and reuse factor: C-simulation shows the result of a setting before you spend
hours building it.

### Train a model

This page uses the jet-tagging model from
[`hls4ml-tutorial`](https://github.com/fastmachinelearning/hls4ml-tutorial)
`1_getting_started/1a_train_keras.ipynb`: five-class classification of jets from 16 features,
about 4,400 parameters.

**Using your own model.** Skip this script and save two files instead:

- the Keras model as `/work/model/model.keras`;
- a float32 test set as `/work/model/X_test.npy`, in exactly the shape your model takes as input
  (for example `(N, 16, 1)` for a Conv1D model, not `(N, 16)`).

Then check these before building:

- The scripts on this page handle **one input and one output**. Models with several inputs or
  outputs need changes to the validation Job.
- Every layer must be supported by hls4ml. `convert.py` in the next step tells you in seconds if one
  isn't.
- Larger models usually need a higher `REUSE_FACTOR` or a lower `PRECISION` to fit on the card; see
  [Settings that fit on the U55C](#check-it).
- Give each model its own name: change `NAME` and `metadata.name` in both Jobs (`jet` on this page),
  so builds don't overwrite each other in `/work/builds/`.

```bash
mkdir -p /work/run /work/model
cat > /work/run/train.py << 'EOF'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

np.random.seed(0)
data = fetch_openml('hls4ml_lhc_jets_hlf')
X, y = data['data'], data['target']
y = np.eye(5)[LabelEncoder().fit_transform(y)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Input(shape=(16,)),
    Dense(64, name='fc1', activation='relu'),
    Dense(32, name='fc2', activation='relu'),
    Dense(32, name='fc3', activation='relu'),
    Dense(5, name='output', activation='softmax'),
])
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=1024, epochs=20, validation_split=0.25, shuffle=True)

model.save('/work/model/model.keras')
np.save('/work/model/X_test.npy', X_test[:1024].astype(np.float32))
print('saved')
EOF

cd /work/run && python3 train.py
```

`fetch_openml` downloads the dataset on first use, so expect a pause before training starts. Twenty
epochs take a few minutes on 2 CPUs.

### Write the conversion script

The build Job in step 3 runs this same script, so what you check here is exactly what gets built.
Settings come from environment variables, with defaults that build on the U55C.

```bash
cat > /work/run/convert.py << 'EOF'
"""Convert /work/model/model.keras with hls4ml and save C-simulation and Keras outputs.
Usage: python3 convert.py PROJECT_DIR RESULTS_DIR"""
import json, os, sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import keras
import hls4ml

s = {
    'precision':    os.environ.get('PRECISION', 'ap_fixed<16,6>'),
    'reuse_factor': int(os.environ.get('REUSE_FACTOR', '4')),
    'strategy':     os.environ.get('STRATEGY', 'Resource'),
    'kernel_slr':   os.environ.get('KERNEL_SLR', 'SLR2'),  # empty string: let Vivado use all three SLRs
}
prj, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

model = keras.models.load_model('/work/model/model.keras', compile=False)
cfg = hls4ml.utils.config_from_keras_model(model, granularity='model', backend='VitisUnified')
cfg['Model']['Precision'] = s['precision']
cfg['Model']['ReuseFactor'] = s['reuse_factor']
cfg['Model']['Strategy'] = s['strategy']

hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, backend='VitisUnified', io_type='io_stream',
    board='alveo-u55c', driver='xrt', axi_mode='axi_master',
    kernel_slr=s['kernel_slr'] or False, clock_period=3.333, clock_uncertainty='27%',
    output_dir=prj, project_name='myproject')
hls_model.write()
hls_model.compile()

X = np.ascontiguousarray(np.load('/work/model/X_test.npy'), dtype=np.float32)
y_csim = np.asarray(hls_model.predict(X), dtype=np.float32).reshape(len(X), -1)
y_keras = model.predict(X, verbose=0).astype(np.float32).reshape(len(X), -1)
np.save(f'{out}/y_csim.npy', y_csim)
np.save(f'{out}/y_keras.npy', y_keras)
json.dump(s, open(f'{out}/settings.json', 'w'), indent=1)
print('settings:', s)
print('C-simulation vs Keras, same predicted class: %.4f' % (y_csim.argmax(1) == y_keras.argmax(1)).mean())
EOF
```

### Check it

```bash
cd /work/run && python3 convert.py /tmp/check-prj /tmp/check
```

The last line should report about 0.99 for the default settings. A much lower value means the
precision is too narrow for your model, and the card will show the same: the hardware reproduces
C-simulation bit for bit. Try other settings by setting variables in front of the command, for
example `PRECISION='ap_fixed<18,8>' python3 convert.py /tmp/check-prj /tmp/check`.

**Settings that fit on the U55C.** With `KERNEL_SLR=SLR2` the design must fit in one SLR (2,879
DSPs). This model at `REUSE_FACTOR=1` with 16-bit weights needs about 3,365 and won't fit. Dense
layers with the `Latency` strategy, 16-bit weights and reuse factor above 1 usually fail routing
after several hours. `Resource` strategy, or 8-bit weights, avoids that. The defaults above
(16-bit, reuse factor 4, `Resource`) build at about 300 MHz.

**Delete the shell pod now.** It isn't needed again, and an idle pod holds CPU and memory that other
users could have. Your model and scripts are on the volume:

```bash
exit
kubectl delete pod hls4ml-shell -n YOUR-NAMESPACE
```

---

## 3. Build the bitstream

The build runs as a Job, not a bare pod: bare pods get reaped during multi-hour runs (exit 137, no
events), and Coder sessions kill long processes even under `screen`. The Job requests no card.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hls4ml-build-jet
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
        image: gitlab-registry.nrp-nautilus.io/nrp/coder-images/vivado-vitis@sha256:92345d9af99ad2aa2265578bd789a94186949196bebe9a4839600f87bab5cecb
        env:
        - { name: NAME,         value: "jet" }
        - { name: PRECISION,    value: "ap_fixed<16,6>" }
        - { name: REUSE_FACTOR, value: "4" }
        - { name: STRATEGY,     value: "Resource" }
        - { name: KERNEL_SLR,   value: "SLR2" }
        - { name: XILINXD_LICENSE_FILE, value: "2100@xilinxd.xilinx-dev" }
        - { name: PLATFORM_REPO_PATHS,  value: "/opt/xilinx/platforms" }
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -eo pipefail
            OUT=/work/builds/$NAME
            mkdir -p $OUT
            exec > >(tee $OUT/build.log) 2>&1

            source /opt/xilinx/xrt/setup.sh
            source /tools/Xilinx/Vitis/2024.2/settings64.sh
            python3 -m venv --system-site-packages /tmp/venv
            source /tmp/venv/bin/activate
            pip install -q "hls4ml @ git+https://github.com/fastmachinelearning/hls4ml@main" "quantizers<2"

            echo "== 1/4 convert and C-simulation  $(date)"
            python3 /work/run/convert.py /scratch/prj $OUT

            echo "== 2/4 HLS synthesis  $(date)"
            cd /scratch/prj/vitis_workspace/myproject
            with-udev-stub v++ -c --mode hls --config hls_kernel_config_csim.cfg --work_dir vitis_unified_project
            with-udev-stub vitis-run --mode hls --package --config hls_kernel_config_csim.cfg --work_dir vitis_unified_project
            cp $(find vitis_unified_project -name myproject_axi_master_csynth.rpt | head -1) $OUT/

            python3 - << 'PY'
            import glob, json, os, re, zipfile
            import xml.etree.ElementTree as ET
            out = '/work/builds/' + os.environ['NAME']
            # kernel name, needed by the validation Job
            xo = glob.glob('/scratch/prj/vitis_workspace/myproject/**/*.xo', recursive=True)[0]
            z = zipfile.ZipFile(xo)
            kernel = ET.fromstring(z.read([n for n in z.namelist() if n.endswith('kernel.xml')][0])).find('.//kernel')
            json.dump({'kernel': kernel.get('name')}, open(f'{out}/meta.json', 'w'))
            # stop before the link if the design cannot fit in its SLR
            rpt = open(f'{out}/myproject_axi_master_csynth.rpt').read()
            bram, dsp, ff, lut = map(int, re.search(r'\|Total\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|', rpt).groups())
            slr = json.load(open(f'{out}/settings.json'))['kernel_slr']
            print(f'HLS estimate: {dsp} DSP, {lut} LUT, {ff} FF, {bram} BRAM')
            if slr and dsp > 2879:
                raise SystemExit(f'STOP: needs {dsp} DSPs, one SLR has 2879. Raise REUSE_FACTOR, '
                                 'lower the precision, or set KERNEL_SLR to "".')
            PY

            echo "== 3/4 link  $(date)"
            cd /scratch/prj/vitis_workspace/system_link
            eval "with-udev-stub $(grep '^v++ -l' link_system.sh)"

            echo "== 4/4 collect  $(date)"
            cp myproject.xclbin $OUT/
            xclbinutil --info --input myproject.xclbin > $OUT/xclbin_info.txt
            grep -m1 "Achieved Freq" $OUT/xclbin_info.txt || true
            echo "build done  $(date)"
        resources:
          limits:   { memory: 40Gi, cpu: "8", ephemeral-storage: 10Gi }
          requests: { memory: 40Gi, cpu: "8", ephemeral-storage: 10Gi }
        volumeMounts:
        - { name: work,    mountPath: /work }
        - { name: tools,   mountPath: /tools/Xilinx, readOnly: true }
        - { name: scratch, mountPath: /scratch }
      volumes:
      - name: work
        persistentVolumeClaim: { claimName: hls4ml-work }
      - name: tools
        persistentVolumeClaim: { claimName: xilinx-tools, readOnly: true }
      - name: scratch
        emptyDir: {}
```

The Job runs four stages:

1. **Convert and C-simulation:** the same `convert.py` as in step 2.
2. **HLS synthesis:** a few minutes. Afterwards the Job reads the size estimate and stops if the
   design can't fit in its SLR, so you find out in about 15 minutes instead of after a failed link.
3. **Link:** the long stage. Only the `v++` link command runs under `with-udev-stub`.
4. **Collect:** copies the xclbin and its clock into `/work/builds/jet/`.

The Vivado project lives on the node's local disk (`emptyDir`), which is faster than CephFS and is
deleted with the pod. Everything you need afterwards is copied to `/work/builds/NAME/`, including
`build.log`. **40 Gi is enough for any model of this size:** peak memory during the link is set by the
card's platform, not by your model, and stays around 25–31 GiB.

```bash
kubectl apply -f build-job.yaml
kubectl get job hls4ml-build-jet -n YOUR-NAMESPACE
```

### Watch it

```bash
kubectl logs job/hls4ml-build-jet -n YOUR-NAMESPACE | grep -E "^== |HLS estimate|STOP|Run vpl" | tail -5
kubectl top pod -l job-name=hls4ml-build-jet -n YOUR-NAMESPACE
```

During the link, `v++` prints a `Run vpl: Step ...` line every minute or so. The first hour is
`config_hw_runs`, where Vivado synthesizes about 80 platform modules in parallel; `kubectl top`
should show several cores and 10–30 GiB. **If it shows a few millicores for more than 20 minutes,
the link is hung,** see [Troubleshooting](#troubleshooting).

The whole Job took 129 minutes in our test (link 1 h 55 min). Across our runs the link for this
model took 62–170 minutes, depending on the node's CPU. To check
whether the Job has finished:

```bash
kubectl get job hls4ml-build-jet -n YOUR-NAMESPACE
```

```
NAME               STATUS     COMPLETIONS   DURATION   AGE
hls4ml-build-jet   Complete   1/1           2h7m       2h8m
```

`COMPLETIONS` reads `0/1` while the build runs and `1/1` when it has succeeded. `STATUS` shows
`Running`, `Complete` or `Failed`. If it says `Failed`, the reason is at the end of the log:
`kubectl logs job/hls4ml-build-jet -n YOUR-NAMESPACE --tail=30`.

To have your terminal wait and return when the build is done instead (it stops waiting after 4 hours):

```bash
kubectl wait --for=condition=complete job/hls4ml-build-jet -n YOUR-NAMESPACE --timeout=4h
```

When it has finished:

```bash
kubectl logs job/hls4ml-build-jet -n YOUR-NAMESPACE --tail=5
kubectl delete job hls4ml-build-jet -n YOUR-NAMESPACE
```

The log ends with the clock Vitis chose, for example `Achieved Freq: 299.4 MHz`, and `build done`.
Finished Jobs are removed by the cluster after a while, but `/work/builds/jet/build.log` stays.

**Building several designs.** Copy the manifest, change `metadata.name`, `NAME` and the settings, and
apply it. Builds run in parallel, each in its own Job.

---

## 4. Validate on the card

This Job requests one U55C, loads the xclbin, runs the test set, and compares the card's output with
C-simulation and with Keras. It holds the card for about a minute.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hls4ml-validate-jet
  namespace: YOUR-NAMESPACE
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        topology.kubernetes.io/region: us-west
      containers:
      - name: validate
        image: gitlab-registry.nrp-nautilus.io/nrp/coder-images/vivado-vitis@sha256:92345d9af99ad2aa2265578bd789a94186949196bebe9a4839600f87bab5cecb
        env:
        - { name: NAME, value: "jet" }
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -eo pipefail
            source /opt/xilinx/xrt/setup.sh
            xbutil examine | grep -E "xilinx_u55c|Version" || true
            cd /tmp && python3 - << 'PY'
            import json, os, re, time
            import numpy as np, pyxrt

            d = '/work/builds/' + os.environ['NAME']
            X = np.ascontiguousarray(np.load('/work/model/X_test.npy'), dtype=np.float32)
            y_csim = np.load(f'{d}/y_csim.npy')
            y_keras = np.load(f'{d}/y_keras.npy')
            n, n_out = len(X), y_csim.shape[1]

            dev = pyxrt.device(0)
            t = time.perf_counter()
            uuid = dev.load_xclbin(pyxrt.xclbin(f'{d}/myproject.xclbin'))
            load_s = time.perf_counter() - t
            k = pyxrt.kernel(dev, uuid, json.load(open(f'{d}/meta.json'))['kernel'])

            bo_in = pyxrt.bo(dev, X.nbytes, pyxrt.bo.normal, k.group_id(0))
            bo_out = pyxrt.bo(dev, n * n_out * 4, pyxrt.bo.normal, k.group_id(1))
            bo_in.write(X.tobytes(), 0)
            bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            t = time.perf_counter()
            k(bo_in, bo_out, np.uint32(n)).wait()
            run_s = time.perf_counter() - t
            bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            y = np.frombuffer(bo_out.map(), dtype=np.float32, count=n * n_out).reshape(n, n_out).copy()

            clock = re.search(r'Achieved Freq:\s*([\d.]+)', open(f'{d}/xclbin_info.txt').read())
            same = (y.view(np.uint32) == y_csim.view(np.uint32)).all(1)
            print(f'kernel clock               : {clock.group(1) if clock else "?"} MHz')
            print(f'identical to C-simulation  : {same.sum()} of {n} rows')
            print(f'same class as Keras        : {(y.argmax(1) == y_keras.argmax(1)).mean():.4f}')
            print(f'max |card - Keras|         : {np.abs(y - y_keras).max():.4f}')
            print(f'xclbin load                : {load_s:.1f} s')
            print(f'throughput (kernel only)   : {n / run_s:,.0f} inferences/s')
            PY
        resources:
          limits:
            memory: 8Gi
            cpu: "2"
            amd.com/xilinx_u55c_gen3x16_xdma_base_3-0: 1
          requests:
            memory: 8Gi
            cpu: "2"
            amd.com/xilinx_u55c_gen3x16_xdma_base_3-0: 1
        volumeMounts:
        - { name: work, mountPath: /work }
      volumes:
      - name: work
        persistentVolumeClaim: { claimName: hls4ml-work }
```

```bash
kubectl apply -f validate-job.yaml
kubectl get pods -l job-name=hls4ml-validate-jet -n YOUR-NAMESPACE
```

`Pending` means every card is in use; the Job starts when one frees up. Check with
`kubectl describe pod -l job-name=hls4ml-validate-jet -n YOUR-NAMESPACE | tail -3`, where
`Insufficient amd.com/xilinx_u55c...` confirms it's waiting for a card. When the Job completes:

```bash
kubectl logs job/hls4ml-validate-jet -n YOUR-NAMESPACE
kubectl delete job hls4ml-validate-jet -n YOUR-NAMESPACE
```

The card reads float32 input and does the conversion to fixed point itself, so the host only copies
buffers.

### What to expect

Our run of this page, November 2026, default settings, 1024 test inputs:

```
kernel clock               : 299.4 MHz
identical to C-simulation  : 1024 of 1024 rows
same class as Keras        : 0.9902
max |card - Keras|         : 0.1904
xclbin load                : 14.5 s
throughput (kernel only)   : 2,710,450 inferences/s
```

Your clock may differ slightly, since Vitis picks the highest clock the routed design meets.
Throughput counts only the kernel run at batch 1024, not the buffer copies or the xclbin load.

**The first two rows are the check that matters.** Every row identical to C-simulation means the
deployment is correct: the card computes exactly what `predict()` computed in step 2. The difference
from Keras comes from fixed-point precision and is already visible in step 2. To change it, change
the precision there, not anything on the cluster.

---

## Troubleshooting

### The link hangs: `config_hw_runs` for hours, a few millicores of CPU

The link ran without `with-udev-stub`. Vivado created its synthesis runs, but none started. Check
that the `v++ -l` line in the build Job starts with `with-udev-stub`. In older images that don't
have the wrapper, the link crashes instead, with a backtrace through `udev_enumerate_scan_devices`.

### `Inconsistency detected by ld.so: dl-lookup.c ... check_match`

A libudev stub was on `LD_LIBRARY_PATH` for the whole job, including Python. Apply
`with-udev-stub` to `v++` and `vitis-run` only.

### `cannot import name 'get_fixed_quantizer_np' from 'quantizers'`

quantizers 2.0 removed a function this hls4ml version imports. Install `"quantizers<2"`.

### `STOP: needs N DSPs, one SLR has 2879`

The design doesn't fit in the SLR it's pinned to. Raise `REUSE_FACTOR`, use 8-bit weights, or set
`KERNEL_SLR` to `""` so Vivado can spread the design over all three SLRs (often at a lower clock).

### Routing fails after several hours (`VPL 35-2` or `35-3`)

Common for dense layers with `Latency` strategy, 16-bit weights and reuse factor above 1. Use
`STRATEGY=Resource`, or 8-bit weights.

### `AttributeError: module 'hls4ml' has no attribute 'utils'`

A directory named `hls4ml` next to your script shadows the installed package, because Python puts the
script's directory first on its import path. Keep scripts in `/work/run`, away from any hls4ml
checkout.

### `ImportError: fetch_openml requires pandas`

Install it: `pip install pandas`.

### Pod died with exit 137 and no events

A bare pod was reaped. Use a Job for anything long-running.

### `ImagePullBackOff` with `context canceled`

The node had no cached copy of the large image. Keep the `topology.kubernetes.io/region: us-west`
node selector.

### Pod stuck in `ContainerCreating` with `MountVolume.MountDevice failed ... DeadlineExceeded`

The node couldn't mount the CephFS volume. Delete the pod and create it again. If it keeps failing on
several nodes, report it on [Nautilus Support](https://nrp.ai/contact/) with the PVC name and the
error.

### Permission denied writing to `/work`

CephFS ignores `fsGroup`. Run the one-time `chown` from [Create a work volume](#create-a-work-volume).

### `Cannot checkout license`

Check that `XILINXD_LICENSE_FILE=2100@xilinxd.xilinx-dev` is set in the environment where `v++` runs.
Test with `lmutil lmstat -c "$XILINXD_LICENSE_FILE"`. See also
[Using Vivado/Vitis with the cluster license server](/documentation/userdocs/fpgas/using-fpgas-from-pods#using-vivadovitis-with-the-cluster-license-server).

### `settings64.sh: No such file or directory`

The tools volume's contents change over time. Run `ls /tools/Xilinx/Vitis/` in a pod that mounts it
and use the version that's there; only 2024.2 has been tested with this page.
