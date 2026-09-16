---
title: hls4ml on a U55C — C++ host
description: Drive an hls4ml kernel on an Alveo U55C from C++ using XRT's native API, as an alternative to the pyxrt host.
---

This page covers **only the last step** of running hls4ml on an Alveo U55C: driving the
card from a C++ host program using XRT's native API, instead of the Python host.

Everything before this — the PVC, the build pod, training, hls4ml conversion, the kernel
wrapper, `v++ -c` and `v++ -l` — is identical for both hosts and lives on
[hls4ml on a U55C — Python host](/documentation/userdocs/fpgas/hls4ml-u55c-python).
**Follow Parts 1 and 2 there first.** Come back here instead of that page's Part 3 if you
want the C++ host.

## Which host should you use?

| | Python (`pyxrt`) | C++ (XRT native) |
| --- | --- | --- |
| Needs a compiler | No | Yes — `g++`, plus XRT include and library paths |
| Extra dependencies in the FPGA pod | numpy | none |
| Throughput measured | ~1.17–1.29M pred/s | ~1.40M pred/s |
| Output | \<- bit-identical -> | \<- bit-identical -> |

Prefer Python unless you are integrating into a C++ pipeline, or you want the lower
measurement overhead. The throughput gap is Python interpreter overhead around
`run.wait()`, not a hardware difference: both hosts produce byte-identical results from
the same bitstream.

## Prerequisites

A `kernel_wrapper.xclbin` at `/work/prj/build/`, and the test vectors at `/work/model/`,
both produced by Parts 1 and 2 of the
[Python host page](/documentation/userdocs/fpgas/hls4ml-u55c-python).

You also need the three model-dependent constants from `firmware/defines.h`:

```bash
grep -E "input_t|result_t" /work/prj/firmware/defines.h
```

```
typedef nnet::array<ap_fixed<16,6>, 16*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 5*1> result_t;
```

`N_IN` is `input_t::size` (16), `N_OUT` is `result_t::size` (5), and `FRAC` is `W - I` for
the `ap_fixed<W,I>` precision (10). Change these in the host if your model differs.

## 1. Request a card

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

Do not request `xilinx.com/fpga_jtag` — loading an `.xclbin` uses the PCIe-side resource
only. See
[Before you request JTAG](/documentation/userdocs/fpgas/using-fpgas-from-pods#before-you-request-jtag-read-this).

```bash
source /opt/xilinx/xrt/setup.sh
xbutil examine
```

One device should appear with shell `xilinx_u55c_gen3x16_xdma_base_3` and
`Device Ready: Yes`.

## 2. The host program

```bash
cat > /work/prj/host.cpp << 'EOF'
// Minimal XRT host for an hls4ml kernel_wrapper on Alveo U55C.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Must match firmware/defines.h
constexpr int N_IN = 16;            // input_t::size
constexpr int N_OUT = 5;            // result_t::size
constexpr int FRAC = 16 - 6;        // ap_fixed<16,6> -> 10 fractional bits

using fixed_t = int16_t;

// ap_fixed defaults to AP_TRN, so this must truncate toward -inf, not round.
static inline fixed_t to_fixed(double x) {
    const double scaled = std::floor(x * (1 << FRAC));
    return static_cast<fixed_t>(std::min(32767.0, std::max(-32768.0, scaled)));
}

static inline double from_fixed(fixed_t v) {
    return static_cast<double>(v) / (1 << FRAC);
}

static std::vector<double> read_features(const std::string &path, size_t &n_samples) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open input file: " + path);
    std::vector<double> vals;
    double v;
    while (f >> v) vals.push_back(v);
    if (vals.empty() || vals.size() % N_IN != 0)
        throw std::runtime_error("input size is not a multiple of N_IN");
    n_samples = vals.size() / N_IN;
    return vals;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <xclbin> <input.dat> [output.dat] [index_or_bdf]\n";
        return 1;
    }
    const std::string xclbin_path = argv[1];
    const std::string input_path = argv[2];
    const std::string output_path = (argc > 3) ? argv[3] : "hw_results.dat";
    const std::string device_arg = (argc > 4) ? argv[4] : "0";

    try {
        size_t n_samples = 0;
        std::vector<double> features = read_features(input_path, n_samples);
        std::cout << "samples: " << n_samples << "\n";

        xrt::device device = (device_arg.find(':') != std::string::npos)
            ? xrt::device(device_arg)
            : xrt::device(static_cast<unsigned int>(std::stoul(device_arg)));
        std::cout << "device: " << device.get_info<xrt::info::device::name>() << "\n";

        xrt::uuid uuid = device.load_xclbin(xclbin_path);
        xrt::kernel krnl(device, uuid, "kernel_wrapper");

        // kernel_wrapper(batchsize, in, out) -> args 0, 1, 2. group_id() places each
        // buffer in the HBM bank the .cfg 'sp=' lines assigned.
        const size_t in_bytes = n_samples * N_IN * sizeof(fixed_t);
        const size_t out_bytes = n_samples * N_OUT * sizeof(fixed_t);

        xrt::bo bo_in(device, in_bytes, krnl.group_id(1));
        xrt::bo bo_out(device, out_bytes, krnl.group_id(2));

        fixed_t *in_map = bo_in.map<fixed_t *>();
        fixed_t *out_map = bo_out.map<fixed_t *>();

        for (size_t i = 0; i < features.size(); ++i) in_map[i] = to_fixed(features[i]);
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto t0 = std::chrono::high_resolution_clock::now();
        xrt::run run = krnl(static_cast<unsigned int>(n_samples), bo_in, bo_out);
        run.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        std::ofstream out(output_path);
        if (!out) throw std::runtime_error("cannot open output file: " + output_path);
        out.precision(8);
        out << std::fixed;
        for (size_t s = 0; s < n_samples; ++s) {
            for (int j = 0; j < N_OUT; ++j) {
                out << from_fixed(out_map[s * N_OUT + j]);
                if (j + 1 < N_OUT) out << " ";
            }
            out << "\n";
        }
        std::cout << "wrote " << n_samples << " results to " << output_path << "\n";
        std::cout << "kernel time: " << sec << " s\n";
        std::cout << "throughput : " << (n_samples / sec) << " pred/s (kernel only)\n";
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
EOF
```

:::caution
The float-to-fixed conversion must **truncate**, not round. `ap_fixed` defaults to
`AP_TRN`, so assigning a float in the HLS testbench truncates toward −∞. A host that
rounds disagrees with C-simulation on roughly 20% of rows — by up to 0.13 on this model —
without ever changing an argmax, so it survives a spot check of a few rows.
:::

## 3. Build and run

```bash
source /opt/xilinx/xrt/setup.sh
cd /work/prj

g++ -std=c++17 -Wall host.cpp -o host \
  -I${XILINX_XRT}/include -L${XILINX_XRT}/lib \
  -lxrt_coreutil -lpthread -luuid

./host build/kernel_wrapper.xclbin \
       /work/model/tb_input_features.dat \
       hw_results.dat
```

No venv is needed — the host links against XRT directly and reads plain text files.

:::caution
If you edit `host.cpp`, re-run `g++`. Editing the source does not update the binary, and a
stale binary produces results that look mysteriously unchanged. Check the timestamp on
`host` if in doubt.
:::

## 4. Compare against Keras

```bash
python3 -c "
import numpy as np
h = np.loadtxt('/work/prj/hw_results.dat')
k = np.loadtxt('/work/model/tb_output_predictions.dat')
print('max abs diff vs Keras:', np.abs(h-k).max())
print('argmax agreement     :', (h.argmax(1)==k.argmax(1)).mean())
"
```

(`pip install --user numpy` if it is missing — needed for this check only, not for the
host itself.)

Expect, on 1024 test samples:

| Metric | Value |
| --- | --- |
| Max abs difference vs Keras float32 | 0.197 – 0.203 |
| Argmax agreement with Keras | 99.2 – 99.4% |
| Throughput, kernel execution only | ~1.40M predictions/s |

The residual difference from Keras is `ap_fixed<16,6>` quantisation, not anything specific
to running on Nautilus.

Release the card when done:

```bash
exit
kubectl delete pod hls4ml-fpga -n YOUR-NAMESPACE
```

## Troubleshooting

### `fatal error: xrt/xrt_bo.h: No such file or directory`

`XILINX_XRT` is not set. Run `source /opt/xilinx/xrt/setup.sh` first.

### `undefined reference to xrt::device::device(...)`

Missing `-lxrt_coreutil`, or the `-L${XILINX_XRT}/lib` path. Use the full `g++` line
above.

### `host.cpp` changes have no effect

The binary was not rebuilt. Re-run `g++` and check the timestamp on `host`.

### Results disagree with the Python host

Check that `to_fixed` uses `std::floor`, not `std::round`, and that `N_IN`, `N_OUT` and
`FRAC` match `firmware/defines.h`.

### `xbutil examine` reports 0 devices

See
[Requesting FPGAs from a Pod](/documentation/userdocs/fpgas/using-fpgas-from-pods#xbutil-examine-reports-0-devices-found).

For anything upstream of the bitstream, see the troubleshooting section of the
[Python host page](/documentation/userdocs/fpgas/hls4ml-u55c-python#troubleshooting).
