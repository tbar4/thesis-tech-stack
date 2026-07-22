# Firmware first: BIOS and the Windows send-off

**Goal.** Get firmware current and validate the hardware while the shipped Windows install is still around to help, then wipe with confidence.

**Covers.** The MSI BIOS update path; why a prebuilt machine ships with stale firmware; the license-in-firmware quirk that makes the Windows send-off painless; and the preflight hardware validation I run before I erase the only OS the vendor tested.

## Theory

### Why firmware is the first thing, not an afterthought

It is tempting to treat the BIOS as something you only touch when something is broken. On a brand-new prebuilt machine that instinct is exactly backwards, and the reason is a supply-chain timing gap. A system like the MSI Aegis R2 (A2NVV9-2218US) is assembled, imaged, boxed, warehoused, and shipped. The firmware flashed onto the board happened at the earliest point in that chain, often months before the box reaches me. Firmware, meanwhile, is where a lot of the platform's hardware-enablement work lives: memory training for the DDR5 kit, PCIe link stability, CPU microcode, fan and thermal curves, and compatibility fixes for exactly the kind of new silicon this machine has.

The baseline machine is a genuinely new platform on two fronts at once. The CPU is an Intel Core Ultra 9 285 (Arrow Lake), and the GPU is an RTX 5080 (Blackwell). New CPU platforms in particular tend to accumulate a rapid series of early BIOS releases that fix memory compatibility, boot reliability, and microcode issues discovered after launch. The firmware that shipped on my board predates most of those fixes by construction. So updating the BIOS before I do anything else is not superstition; it is me pulling in months of platform fixes that the factory image could not have included.

```admonish gotcha title="Microcode and stability on a fresh Arrow Lake board"
Early Arrow Lake platforms saw several BIOS revisions in their first months addressing memory training and boot stability. If you skip the BIOS update and later chase a flaky-boot or memory-instability ghost on Linux, you can burn a day blaming the NVIDIA driver or the kernel when the actual fix was a firmware release that shipped after your unit was boxed. Update firmware first, precisely so that when something misbehaves later you have already eliminated the single most likely stale-platform cause.
```

### License-in-firmware: why the Windows send-off is clean

There is a small, delightful fact about modern OEM Windows that makes wiping the shipped install stress-free: the Windows license is not a sticker anymore, and it is not a file on the disk. Since the OEM Activation 3.0 scheme, the Windows product key for a prebuilt machine is embedded in the firmware, in an ACPI table called MSDM (Microsoft Digital Marker). The key lives in the BIOS. That means I can erase the entire drive, install Linux, and the Windows entitlement is still sitting safely in firmware. If I ever need Windows back on this hardware (for a firmware tool that only ships as a Windows binary, say), a fresh Windows installer reads the MSDM key from the board and activates automatically, no key entry, no phone call.

You can see the key before you wipe, which is a nice way to confirm the machine is genuinely licensed and that the send-off will be clean. From the shipped Windows, an elevated PowerShell prompt reports it:

```powershell title="powershell - read the firmware-embedded Windows key (run in shipped Windows)"
(Get-WmiObject -Query 'SELECT * FROM SoftwareLicensingService').OA3xOriginalProductKey
```

If that returns a key, the MSDM table is populated and the entitlement will survive the wipe. Later, from Linux, you can confirm the same table is present without any Windows at all:

```bash title="bash - confirm the MSDM (Windows license) table exists, from Linux"
# The MSDM ACPI table is what holds the OEM Windows key in firmware.
sudo test -f /sys/firmware/acpi/tables/MSDM && echo "MSDM present (Windows entitlement lives in firmware)" || echo "no MSDM table"
```

The practical upshot: I do not need to hoard a product key, and I do not need to feel precious about the shipped OS. The license is in the board. I keep the shipped Windows just long enough to do the firmware update and the burn-in, then I erase it without a second thought.

### Why validate hardware before wiping

Here is the reasoning that gives this chapter its shape. The shipped Windows install is the only software configuration the vendor actually tested against this exact hardware. Its drivers are matched, its diagnostic tools run natively, and if a component is dead on arrival, the vendor's own utilities will say so in language their support desk understands. The moment I wipe it, I take on the burden of proving that every hardware fault is not my fault. So I do my destructive-testing and my burn-in while the vendor's known-good environment is still on the disk. If the memory is bad, the GPU is unstable, or a fan does not spin, I want to find out now, with a clean RMA story ("your shipped OS, your diagnostic tool, your fault"), not three weeks into a Linux setup where support will reasonably ask whether I broke it myself.

There is also a firmware-update safety argument for keeping Windows briefly. MSI's most convenient BIOS-flashing tools historically run as Windows applications, and while the board also supports flashing from a USB stick via the BIOS itself (M-Flash), doing the first update from the vendor's own environment removes a variable. So the order is deliberate: update firmware from the tested OS, burn in on the tested OS, capture the preflight results, and only then erase.

### What a burn-in is actually looking for

It helps to be precise about what sustained-load testing catches, because "stress test" is a vague phrase that hides three distinct failure modes. The first is **infant mortality**: components that are marginal from the factory tend to fail early, under their first real thermal cycling, rather than randomly across their lifetime. A half-hour of full load is a compressed version of that early stress, so if a solder joint, a fan bearing, or a VRM is going to die young, I would much rather provoke it now, inside the warranty window and on the vendor's OS, than three weeks into a training run. The second is **thermal design error**: a heatsink that was not seated flat, a fan curve that ramps too late, or case airflow that recirculates hot air will not show up in a five-second check but will show up as a temperature that keeps climbing toward the throttle point under sustained load. The third is **power delivery instability**: a PSU that sags under combined CPU-plus-GPU load, which manifests as a crash or reset only when both are hammered at once, which is why I run a brief combined-load pass and not just each component alone.

None of these three is a correctness problem, which is the important distinction from the memory test. MemTest86 is looking for silent wrongness (a bit that flips and corrupts data with no crash). The burn-in is looking for instability and heat (a machine that crashes, throttles, or degrades under load). A thorough preflight needs both, because they fail differently and a machine can pass one while failing the other. A GPU with perfect memory can still thermal-throttle a long training run into uselessness, and a perfectly cool machine can still have one bad DRAM cell that quietly poisons a checkpoint.

```admonish under-the-hood title="Why factory 'tested' is not the same as burned-in"
A prebuilt vendor does run functional tests, but those are fast pass/fail checks designed for throughput on an assembly line, not multi-hour thermal soaks on your specific unit in your specific room at your specific altitude and ambient temperature. Their test proves the machine POSTs and the parts respond; it does not prove your unit holds clocks at your desk after twenty minutes. The gap between 'it powered on and passed QA' and 'it sustains full load without throttling in my environment' is exactly the gap a personal burn-in closes, which is why I do it even on a machine that arrived with a passing factory sticker.
```

## Tooling

Three tools carry this chapter: MSI's firmware update path, a memory tester, and a GPU/CPU stress harness. I will name the actual surfaces here and drive them in the Lab.

**MSI BIOS updates** come from the product's own support page. Every MSI board reports its exact model and current BIOS version, and the support page lists BIOS releases with dates and per-release changelogs. There are two ways to apply one. M-Flash reboots into the BIOS and flashes from a FAT32 USB stick holding the BIOS file, and it is the method I trust for the actual flash because it does not depend on a running OS. MSI Center (Windows) can also fetch and apply updates, which is convenient for discovering that an update exists. My rule: discover the version any way, flash with M-Flash.

You can read the current firmware version from either OS. From Linux:

```bash title="bash - report current firmware version and board model"
sudo dmidecode -s bios-version
sudo dmidecode -s bios-release-date
sudo dmidecode -s baseboard-product-name
sudo dmidecode -s system-product-name   # should report the Aegis R2 SKU
```

**MemTest86** is the memory tester. Bad or mistrained DDR5 is the single most insidious fault for this kind of work, because it produces silent corruption, not clean crashes: a flipped bit in a weight tensor or a training batch just makes your numbers subtly wrong. MemTest86 boots from its own USB stick, outside any OS, and walks the full address space with a battery of patterns. It is the one test I never skip.

**Stress and burn-in** is about heat and sustained load, not correctness. A prebuilt machine can pass a five-second check and then throttle or crash under twenty minutes of real load because a heatsink is poorly seated or a fan curve is wrong. On the shipped Windows I use FurMark for the GPU and Prime95 (or MSI's built-in stress tooling) for the CPU, watching temperatures and clocks. On Linux later I will re-run an equivalent sustained-load test as part of acceptance (Chapter 0.7); here the goal is only to shake out infant mortality while the warranty environment is intact.

```admonish under-the-hood title="What M-Flash actually does, and why you do not interrupt it"
M-Flash erases the SPI flash chip on the motherboard that holds the UEFI firmware and writes the new image in its place. During the write, the board is running from a small resident stub, not from a complete firmware. If power is lost mid-write, the flash can be left half-written and the board will not POST, which is the classic "bricked BIOS" failure. This is why you flash from a stable power state, why you do not touch the machine while it runs, and why laptops refuse to flash on battery. The Aegis R2, like most modern MSI boards, has Flash BIOS Button hardware recovery that can reflash from USB even with no CPU or RAM installed, which is a strong safety net, but the discipline is the same: start the flash, then leave it completely alone until it reboots itself.
```

## Lab

The artifact for this chapter is a filled-in preflight checklist committed alongside the machine log. It is the document I would show a warranty desk, and later the first page of the machine's provenance. I run the whole sequence from the shipped Windows (for the flash and the stress tests) plus a couple of confirmations from a Linux live USB, and I record every result with a value, a date, and the firmware version in effect.

### Step 1 - record the as-shipped state

Before changing anything, capture what arrived. Boot a Linux live USB (or use the shipped Windows) and read the identifiers:

```bash title="scripts/preflight-record.sh"
#!/usr/bin/env bash
# Capture the as-shipped hardware and firmware identity.
# Run from a Linux live USB or the shipped OS. Output is the top of the preflight log.
set -euo pipefail

OUT="preflight-$(date +%Y%m%d).log"
{
  echo "# Preflight record - $(date --iso-8601=seconds)"
  echo
  echo "## Firmware"
  echo "BIOS version : $(sudo dmidecode -s bios-version)"
  echo "BIOS date    : $(sudo dmidecode -s bios-release-date)"
  echo "Board        : $(sudo dmidecode -s baseboard-product-name)"
  echo "System       : $(sudo dmidecode -s system-product-name)"
  echo
  echo "## CPU"
  lscpu | grep -E 'Model name|Socket|Core\(s\)|Thread'
  echo
  echo "## Memory"
  sudo dmidecode -t memory | grep -E 'Size|Speed|Manufacturer|Part Number' | grep -v 'No Module'
  echo
  echo "## GPU (PCI view; driver not required)"
  lspci | grep -Ei 'vga|3d|display'
  echo
  echo "## Windows entitlement"
  if sudo test -f /sys/firmware/acpi/tables/MSDM; then
    echo "MSDM present - Windows license lives in firmware, safe to wipe"
  else
    echo "MSDM absent - investigate before wiping"
  fi
} | tee "$OUT"
echo "Wrote $OUT"
```

### Step 2 - update the BIOS

Find the current BIOS version (Step 1) and compare it to the latest release on the MSI product support page for this exact SKU. If the board is behind, download the latest BIOS, unzip it to a freshly FAT32-formatted USB stick, and flash with M-Flash:

```text title="M-Flash procedure (do not interrupt power)"
1. Copy the unzipped BIOS file to the root of a FAT32 USB stick.
2. Reboot; press Del to enter BIOS setup.
3. Select M-Flash; confirm the reboot into the flash utility.
4. Choose the BIOS file on the USB stick; confirm.
5. Wait. The board writes flash and reboots itself, possibly twice.
   Do NOT power off, do NOT press keys until it returns to POST.
6. Re-enter BIOS; confirm the new version string; load optimized defaults.
```

Record the before and after versions in the log. This is the single most important line in the whole preflight, because it timestamps the platform state that every later measurement in the book assumes.

### Step 3 - memory test

Boot MemTest86 from its own USB stick and run at least one full pass (all tests). A full pass on 32 GB of DDR5 takes a while (the exact wall-clock is a machine-log entry; measured on the baseline machine; record value, date, firmware). Zero errors is the only acceptable result. A single error means RMA the memory now, before it silently corrupts a checkpoint later.

### Step 4 - sustained-load burn-in

From the shipped Windows, run the GPU and CPU under sustained load and watch temperatures and clocks. The purpose is to catch throttling and infant mortality, not to benchmark:

```text title="Burn-in procedure (shipped Windows, ~30 min each)"
GPU  : FurMark, stress preset, ~20-30 min. Watch for artifacts,
       driver resets, or thermal throttling. Log peak GPU temp (°C)
       and whether clocks held.
CPU  : Prime95 (Small FFTs) or MSI's stress tool, ~20-30 min.
       Log peak package temp (°C) and whether any core threw errors.
Both : run together briefly to check the PSU and case airflow under
       combined load.
```

All temperatures and clocks here are machine-log entries; measured on the baseline machine; record value, date, firmware. The pass condition is qualitative and strict: no crashes, no artifacts, no driver resets, and temperatures that plateau below throttling rather than climbing without bound.

### Step 5 - fill and commit the checklist

The artifact is this filled-in checklist, saved as `preflight-<date>.md` next to the log from Step 1 and committed to the machine-log repository (the same repo Chapter 0.3 uses for the `nvidia-smi` capture):

```markdown title="preflight-YYYYMMDD.md (the artifact - fill every value)"
# Preflight checklist - MSI Aegis R2 (A2NVV9-2218US)

- Date: ____________________    Operator: ____________________
- BIOS before: __________  →  BIOS after: __________  (release date: ______)
- CPU reported: Intel Core Ultra 9 285              [ ] confirmed
- GPU reported: NVIDIA RTX 5080 16GB                [ ] confirmed
- Memory: 32 GB DDR5, ____ MT/s, all slots seen     [ ] confirmed
- NVMe: 1 TB present, SMART healthy                 [ ] confirmed
- MSDM present (Windows license in firmware)        [ ] confirmed
- MemTest86: ____ full pass(es), errors: ____        [ ] PASS (0 errors)
- GPU burn-in: ____ min, peak ____ °C, clocks held?  [ ] PASS
- CPU burn-in: ____ min, peak ____ °C, errors: ____  [ ] PASS
- Combined load: stable? ____                        [ ] PASS
- Decision: [ ] proceed to wipe   [ ] RMA - reason: ______________
```

**What you should see.** By the end of this sequence you should be holding a single committed document that says, in effect: the firmware is current as of a named release and date, the memory passed a full MemTest86 sweep with zero errors, and the GPU and CPU held stable temperatures under half an hour of sustained load each. Every blank is filled with a real value and a real date. The Windows entitlement is confirmed to live in firmware, so the shipped OS is now expendable. That document is your license to wipe: from here on, every problem you hit is a software problem on a hardware base you have personally proven, and the next chapter can erase Windows without a shred of anxiety.

```admonish thesis-thread
For the committee, the preflight checklist is the provenance root of the entire measurement chain. Every performance number later in the thesis is implicitly conditioned on "hardware that passed preflight on date D at firmware version V." Committing this document, rather than trusting memory, is what lets a reader distinguish a genuine result from an artifact of faulty RAM or a throttling GPU. It is cheap insurance against the most embarrassing possible failure: a beautiful reasoning-delta curve that turns out to be a bad memory module.
```

```admonish substack-seed
There is a satisfying post in the phrase "the license is in the board." Most people still picture a Windows key as a sticker or a slip of paper, and the reveal that a prebuilt machine carries its entitlement in an ACPI firmware table (MSDM) reframes the whole ritual of wiping a new PC: you are not throwing away something precious, you are freeing hardware whose license can never actually leave it. Pair that with the counterintuitive advice to stress-test before you wipe, not after, and you have a tidy piece about doing destructive things carefully: prove the machine with the vendor's own known-good tools while you still can, so that every later fault has a clean owner.
```
