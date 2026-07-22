# Ubuntu 24.04 on Blackwell

**Goal.** Install Ubuntu 24.04 with a kernel and driver that actually drive Blackwell and Arrow Lake, and prove the GPU link is what the spec promises.

**Covers.** Point-release and HWE kernels, and why a recent kernel matters for Arrow Lake and the Wi-Fi module; Secure Boot and MOK enrollment; `nomodeset` recovery; the NVIDIA 570-open driver and why Blackwell requires the open kernel modules; and verifying the PCIe Gen5 x16 link.

## Theory

### Why the kernel version is the first Linux decision

On older, well-worn hardware you can install more or less any Ubuntu and it just works, because the kernel that ships in the installer already contains drivers for silicon that has been in the wild for years. The baseline machine is the opposite case. Both the CPU (Core Ultra 9 285, Arrow Lake) and the platform around it (the Wi-Fi module, the chipset, the integrated graphics on the CPU) are new enough that hardware enablement for them landed in the Linux kernel only recently. Install too old a kernel and the symptoms are exactly the annoying ones: no Wi-Fi, flaky suspend, missing sensors, an integrated GPU the kernel does not recognize.

Ubuntu solves this with two mechanisms, and it is worth understanding both because they interact. The first is **point releases**: 24.04 gets periodic refreshes (24.04.1, 24.04.2, and so on), and each point release rolls in a newer kernel and updated installer media. Installing from the latest point-release ISO, rather than the original 24.04.0 media, is the single cheapest way to start on a kernel that already knows about your hardware. The second is the **HWE (Hardware Enablement) stack**: an LTS release can track a newer "hardware enablement" kernel backported from a later Ubuntu, so that a 24.04 LTS machine can run a kernel several versions ahead of the one it launched with while keeping the LTS userland. For new silicon, HWE is the difference between "supported eventually" and "supported now."

```admonish gotcha title="Old install media on Arrow Lake: no Wi-Fi, and worse"
If you install from the original 24.04.0 ISO on this platform, do not be surprised if the Wi-Fi module is invisible and the installer cannot reach the network to fetch fixes - a chicken-and-egg trap. The fix is to install from the latest 24.04 point-release ISO (which carries a newer kernel) and to have a wired connection or a USB Ethernet/Wi-Fi dongle on hand for the first boot, so you can pull the HWE kernel and updates even if the built-in Wi-Fi is not yet happy. Confirm your running kernel afterward with `uname -r` and make sure it is the HWE line, not the base GA line.
```

The reasoning that follows from this: I install from the newest point-release media I can get, I make sure the HWE kernel meta-package is installed so that future kernel updates track hardware enablement, and I keep a wired fallback for the first boot. A recent kernel is not a nice-to-have on this machine; it is the precondition for the machine being fully usable at all.

### Secure Boot, MOK, and signed kernel modules

Modern Ubuntu ships with Secure Boot enabled, and Secure Boot's whole job is to refuse to load kernel code that is not signed by a key the firmware trusts. That is a good default, but it collides with the fact that the NVIDIA driver is an out-of-tree kernel module that has to be compiled and inserted on your machine. A freshly built NVIDIA module is not signed by anything the firmware knows about, so with Secure Boot on, the kernel will refuse to load it, and you get a system that boots to a black screen or falls back to software rendering with no NVIDIA driver in sight.

Ubuntu's answer is MOK, the Machine Owner Key. When the driver is installed via DKMS, Ubuntu generates a local signing key, signs the module with it, and asks you to **enroll** that key into the firmware's trust store on the next reboot, through a blue MOK-management screen that appears before the OS loads. You confirm enrollment with a password you set during install, and from then on the firmware trusts modules signed by your local key. It is a one-time dance per key, and skipping it (or clicking through it wrong) is the most common reason a correctly installed NVIDIA driver still will not load.

```admonish under-the-hood title="What MOK enrollment actually changes"
Secure Boot verifies a signature chain rooted in keys held in firmware NVRAM: Microsoft's keys, the OEM's, and - after enrollment - yours. MOK enrollment appends your locally generated public key to the Machine Owner Key list, which the shim bootloader consults in addition to the firmware db. After that, any module signed by the matching private key (which DKMS uses automatically on every kernel or driver rebuild) passes verification. This is why you enroll once but never have to re-sign manually: DKMS re-signs on each rebuild with the same enrolled key. If you ever reset Secure Boot keys in the BIOS, you invalidate the enrollment and must repeat the dance. You can check the current state any time with `mokutil --sb-state` and list enrolled keys with `mokutil --list-enrolled`.
```

### nomodeset: the recovery hatch

There is a failure mode worth pre-loading into your head before it happens, because it happens at the worst time: you finish the install, reboot, and get a black screen or an unusable low-resolution desktop, because the graphics stack tried to mode-set on a GPU whose driver is not ready. The escape hatch is a kernel boot parameter, `nomodeset`, which tells the kernel not to hand graphics mode-setting to the GPU driver at boot, falling back to a generic framebuffer. That gets you a working (if ugly) desktop or console from which you can fix the real driver situation.

You reach it from the GRUB menu at boot: highlight the Ubuntu entry, press `e`, find the `linux` line, and append `nomodeset` before booting. It is temporary (it does not survive a reboot unless you edit the GRUB config), which is exactly what you want: use it to get in, fix the driver, and reboot back to a normal graphical boot. On a Blackwell card especially, where the correct driver is non-negotiable (see below), `nomodeset` is the difference between "recoverable" and "reinstall from USB."

### Why Blackwell requires the open kernel modules

Here is the piece that is specific to this generation and trips people up. NVIDIA ships its Linux driver in two kernel-module flavors: the long-standing **proprietary** modules, and the newer **open** modules (often written "open" or "-open", e.g. `nvidia-driver-570-open`). "Open" here refers to the kernel-module source being open; the userspace CUDA stack is unchanged. For older GPUs you could choose either. For Blackwell you cannot: the RTX 5080 and its GB20x-generation siblings are supported **only** by the open kernel modules. The proprietary modules do not support these GPUs at all. So on the baseline machine, "NVIDIA 570-open" is not a preference, it is the only thing that will drive the card.

The reason is architectural. NVIDIA moved the driver's hardware-management logic into a GPU System Processor (GSP) firmware blob that runs on the card itself, and the open kernel modules are built around that GSP-firmware model. Newer architectures (Turing onward, and mandatorily on the latest generations) are designed for this split, which is why the open modules are the supported path for them and the sole path for Blackwell. Practically: I install `nvidia-driver-570-open` (the 570 branch is the one that carries Blackwell support), and if I ever see advice to install the plain `nvidia-driver-570` proprietary variant on this card, I ignore it, because it will not load.

```admonish gotcha title="Installing the proprietary 570 on Blackwell = black screen"
If you install the non-open `nvidia-driver-570` (or an `ubuntu-drivers autoinstall` that happens to pick the proprietary flavor) on the RTX 5080, the modules will build but the GPU will not come up - you get a black screen or a fallback to the iGPU, and `nvidia-smi` reports no devices. The fix is to purge it and install the `-open` variant explicitly. Always confirm with `nvidia-smi` that the card is actually detected before you conclude the driver "installed"; a successful `apt install` is not proof the module drives the hardware.
```

### Verifying the Gen5 x16 link

The last piece of theory is a habit: never assume the GPU negotiated the link you paid for. The RTX 5080 sits in a PCIe Gen5 x16 slot, and a Gen5 x16 link is a lot of bandwidth. But a reseated card, a slot that shares lanes with an NVMe drive, a BIOS setting, or a marginal riser can silently negotiate down to x8, or to Gen4, and nothing will crash. The only symptom is throughput that is quietly lower than it should be, which is precisely the kind of bug that poisons benchmarks. So verifying the link speed and width after install is a standing acceptance step, and I capture the result into the machine log the same way I capture `nvidia-smi`.

### Why apt and DKMS instead of NVIDIA's .run installer

There are two ways to put the NVIDIA driver on a Linux box, and the choice matters more on this stack than it looks. NVIDIA ships a self-contained `.run` installer that compiles and installs the driver directly, and Ubuntu ships the same driver as `apt` packages that build the kernel module through DKMS (Dynamic Kernel Module Support). I use the `apt`/DKMS path, and the reason is the interaction between three moving parts: the HWE kernel (which updates), Secure Boot (which demands signed modules), and the driver (which is an out-of-tree module that must be rebuilt for each kernel).

DKMS automates exactly that interaction. When the HWE kernel updates (and on this machine it will, because hardware enablement is still landing), DKMS automatically rebuilds the NVIDIA module against the new kernel and re-signs it with the enrolled MOK key, so the driver keeps working across kernel upgrades with no manual intervention. The `.run` installer does none of that: it builds the module once against the current kernel, and the next kernel update leaves you with a driver that will not load, at which point you are re-running the installer from a text console. On a Secure Boot machine the `.run` path is worse still, because you have to arrange module signing yourself. So the packaged path is not just more convenient, it is the one that composes correctly with the two things this chapter already committed to: a kernel that moves and a Secure Boot chain that must stay signed.

```admonish gotcha title="Mixing the .run installer and apt drivers leaves a mess"
If you ever install the NVIDIA `.run` driver and later try to install the `apt` driver (or vice versa) without fully removing the first, you can end up with two half-installed drivers fighting over the same files, which produces exactly the black-screen-on-boot symptom that is maddening to diagnose. Pick one path and stay on it. On this stack the path is `apt install nvidia-driver-570-open`; if you ever used the `.run` installer, run its `--uninstall` before switching, and confirm with `dpkg -l | grep nvidia` that the package state is clean.
```

## Tooling

The tools here are the Ubuntu installer, `apt` plus `ubuntu-drivers`, `mokutil`, and `nvidia-smi` and `lspci` for verification.

**The installer** is Ubuntu Desktop 24.04, latest point release. I use the normal graphical installer, choose to erase the disk (the shipped Windows is already validated and expendable, per Chapter 0.2), and set a Secure Boot / MOK password when prompted so third-party drivers can be signed and enrolled.

**Driver installation** goes through Ubuntu's packaging rather than NVIDIA's `.run` installer, because `apt` + DKMS handle the MOK signing and the kernel-rebuild-on-update automatically. `ubuntu-drivers devices` lists the recommended driver; I install the `-open` package explicitly rather than trusting autoinstall to pick the flavor.

**`mokutil`** inspects and manages Secure Boot and MOK state. **`nvidia-smi`** is the ground truth that the driver is actually driving the card. **`lspci -vv`** reports the negotiated PCIe link speed and width.

```admonish read-along title="Names to keep straight"
"570" is the driver branch (the version series that carries Blackwell support). "-open" is the kernel-module flavor (open-source kernel modules, mandatory for Blackwell). "HWE" is the hardware-enablement kernel line for the 24.04 LTS. "MOK" is your local module-signing key for Secure Boot. None of these are interchangeable, and mixing them up is the source of most install-day pain. When in doubt: 570-open kernel modules, HWE kernel, MOK enrolled.
```

## Lab

The full sequence is: install Ubuntu from current media, get onto the HWE kernel, install the 570-open driver, enroll MOK, reboot, then capture `nvidia-smi` and the PCIe link speed and commit both to a machine-log repository. The artifact is that committed capture.

### Step 1 - install and reach the HWE kernel

Install Ubuntu Desktop 24.04 (latest point release), erasing the disk. On first boot, get online (wired or a USB dongle if built-in Wi-Fi is not yet happy), then update and ensure the HWE kernel stack is installed:

```bash title="scripts/01-base-kernel.sh"
#!/usr/bin/env bash
# Get fully updated and onto the HWE (hardware-enablement) kernel line.
set -euo pipefail

sudo apt update && sudo apt full-upgrade -y

# HWE kernel + graphics HWE meta-packages for 24.04.
sudo apt install -y \
  linux-generic-hwe-24.04 \
  linux-image-generic-hwe-24.04 \
  build-essential dkms mokutil pciutils

echo "Reboot, then confirm the running kernel:"
echo "  uname -r    # expect the HWE kernel, not the base GA kernel"
```

Reboot and confirm with `uname -r` that you are on the HWE kernel line.

### Step 2 - install the 570-open driver

Check what Ubuntu recommends, then install the open flavor explicitly:

```bash title="scripts/02-nvidia-open.sh"
#!/usr/bin/env bash
# Install the NVIDIA 570 OPEN kernel modules - mandatory for Blackwell (RTX 5080).
set -euo pipefail

# See what Ubuntu detects/recommends (informational).
ubuntu-drivers devices || true

# Install the OPEN 570 driver explicitly. Do NOT install the non-open variant on Blackwell.
sudo apt update
sudo apt install -y nvidia-driver-570-open

echo
echo "During install, DKMS signs the module with a local MOK key."
echo "On the NEXT reboot you MUST complete MOK enrollment:"
echo "  - a blue 'Perform MOK management' screen appears BEFORE the OS loads"
echo "  - choose 'Enroll MOK' -> Continue -> Yes -> enter the password you set"
echo "  - the machine reboots and the module can now load under Secure Boot"
```

### Step 3 - enroll MOK and reboot

Reboot. Before the OS loads, the blue MOK-management screen appears. Choose Enroll MOK, Continue, Yes, and enter the password. The machine reboots into a normal graphical session with the NVIDIA driver loaded. Confirm Secure Boot and enrollment state:

```bash title="bash - confirm Secure Boot is on and the module is trusted"
mokutil --sb-state          # expect: SecureBoot enabled
mokutil --list-enrolled | grep -i -A2 'nvidia\|DKMS\|Machine Owner' || true
```

If the driver still will not load, this is where `nomodeset` earns its keep: reboot, edit the GRUB `linux` line to append `nomodeset`, boot to a working desktop, and re-check the driver install and MOK state from there.

### Step 4 - capture the proof and commit it

Now produce the artifact. This script captures the driver state and the negotiated PCIe link and writes a dated log for the machine-log repository:

```bash title="scripts/03-capture-gpu-state.sh"
#!/usr/bin/env bash
# Capture nvidia-smi + PCIe link speed/width as the machine-log artifact for this install.
set -euo pipefail

STAMP="$(date --iso-8601=seconds)"
OUT="gpu-state-$(date +%Y%m%d).log"

# Find the NVIDIA GPU's PCI address for the link query.
GPU_ADDR="$(lspci | grep -Ei 'nvidia.*(vga|3d)' | head -n1 | awk '{print $1}')"

{
  echo "# GPU state capture - $STAMP"
  echo "kernel        : $(uname -r)"
  echo "secure boot   : $(mokutil --sb-state 2>/dev/null || echo 'unknown')"
  echo
  echo "## nvidia-smi"
  nvidia-smi
  echo
  echo "## Driver / CUDA versions"
  nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv
  echo
  echo "## PCIe link (negotiated) for $GPU_ADDR"
  # LnkSta is the negotiated state; expect Speed 32GT/s (Gen5), Width x16.
  sudo lspci -vv -s "$GPU_ADDR" | grep -E 'LnkCap|LnkSta'
  echo
  echo "## PCIe link (as nvidia-smi sees it)"
  nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv
} | tee "$OUT"

echo "Wrote $OUT - commit this to the machine-log repo."
```

Commit the result to a small, dedicated machine-log git repository (the same one that holds the Chapter 0.2 preflight checklist). That repo becomes the machine's permanent provenance record: firmware version, install date, driver version, and link speed, all timestamped.

```bash title="bash - commit the capture to the machine-log repo"
cd ~/machine-log
cp ~/gpu-state-*.log ./
git add gpu-state-*.log preflight-*.md
git commit -m "Install capture: 570-open driver + PCIe link on Aegis R2"
```

**What you should see.** The `nvidia-smi` table should print with the card named as an RTX 5080, 16 GB (16384 MiB) total memory, and a driver version on the 570 branch. Secure Boot should report enabled, and the driver should still load, which together prove MOK enrollment worked. The PCIe capture should show the negotiated link at Gen5 (32 GT/s) and width x16 in both the `lspci` `LnkSta` line and the `nvidia-smi` query; if either reads x8 or Gen4, stop and investigate the slot, the BIOS, and the card seating before trusting any throughput number later. The concrete values (exact driver build, temperatures, clocks) are machine-log entries; measured on the baseline machine; record value, date, driver. The committed `gpu-state-*.log` is the artifact: from here on, any performance claim in the book can be traced to a machine whose driver and link speed are on the record.

```admonish thesis-thread
For the committee, this capture pins two of the five things every run logs (Chapter 0.6): the driver version and the hardware link state. A reasoning-delta number is only comparable across time if the driver did not silently change and the GPU did not silently negotiate down. Committing the `nvidia-smi` and PCIe capture at install, and re-checking it during acceptance, is what lets the thesis assert that a before/after comparison ran on the same physical substrate rather than on a card that quietly halved its bandwidth between runs.
```

```admonish substack-seed
The sharp, shareable idea here is "on this GPU there is only one right driver, and it is the open one." Most people still carry the folk knowledge that NVIDIA's proprietary driver is the real one and the open modules are a lesser open-source alternative. Blackwell flips that: the open kernel modules are mandatory and the proprietary ones simply do not support the card. That inversion is a clean hook for a post about how fast the ground moves in this field, wrapped around the genuinely useful, save-your-evening tips: install from current media so Wi-Fi exists, keep `nomodeset` in your back pocket, and never trust that a GPU negotiated the PCIe link on the box until `lspci` tells you it did.
```
