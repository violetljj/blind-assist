# BlindAssist home redesign QA

## Target and runtime state

- Reference: `C:\Users\26442\.codex\generated_images\01a0302d-4040-7590-abbb-b109c5a550e5\exec-c233b5b7-6d01-4025-8d0e-e63dbc660990.png`
- Final implementation: `F:\ba-data\blindassist-artifacts-20260805\ui-redesign-20260824\implementation\home-v5\20260824T045500Z-emulator-5564-screen.png`
- Combined comparison: `F:\ba-data\blindassist-artifacts-20260805\ui-redesign-20260824\design-qa\source-vs-home-v5.png`
- State: returning user, Chinese, Daily mode selected, glasses disconnected, Assist tab selected.
- Device: Android 15 API 35 emulator, 1080 x 2400 px, density 420.
- Normalization: the Android status and navigation bars were excluded by cropping the 1080 x 2208 app viewport at y=72, then resizing it to 852 x 1844 for comparison with the 853 x 1844 reference. System chrome remains visible in the unmodified implementation capture.

## Comparison history

1. v1 exposed light content under light system icons and allowed the bottom surface to consume the system navigation area. Edge-to-edge system-bar appearance and navigation insets fixed both P2 issues.
2. v2 still placed the header, hero, selector, and lower information rows too far from their reference positions. Spacing, type scale, component height, and icon treatment were aligned in v3.
3. v3 used a greener CTA and a stronger sage background wash than the reference. The final v4 uses a darker matte forest gradient and lower-saturation ivory, sage, and blue light fields.
4. v5 replaced filled utility glyphs with the closest available outlined Material variants, bringing the CTA and information rows closer to the reference.
5. The v5 source/implementation pair was inspected at the normalized full-screen size. No P1 or P2 visual mismatch remains.

## Fidelity surfaces

- Typography: brand, readiness state, hero, CTA hierarchy, mode labels, information rows, and bottom-navigation labels match the reference hierarchy and weight. Android's installed CJK font is retained for reliable native rendering.
- Spacing and geometry: horizontal margins, hero rhythm, CTA radius/height, selector proportions, dividers, information-row spacing, and floating bottom navigation align with the reference. The native bottom navigation sits above the Android gesture/navigation inset by design.
- Color and material: the page uses a warm ivory spatial field with restrained cool and sage depth; cards use matte forest green or soft elevated white surfaces; borders and shadows remain low contrast.
- Assets and icons: the screen contains no raster illustration asset. Standard Material icons are used at native density. The available library's visibility and shield glyphs are the closest semantic substitutes for the reference's glasses and shield-lock glyphs.
- Copy and state: all visible Chinese copy matches the selected design intent and is backed by live app state. Daily, Quiet, and Sensitive modes, the primary assistance CTA, and Assist/Settings navigation are functional.

## Accessibility and responsive checks

- Interactive mode changes and Assist/Settings navigation passed connected Compose tests.
- The home screen was checked at system font scale 1.5. The CTA copy reflows without clipping, and a scroll exposes both device and local-processing rows while the bottom navigation remains reachable.
- Edge-to-edge contrast was inspected with dark system icons over the light home surface.

## Residual differences

- P3: glasses and shield-lock use the closest available standard Material glyphs rather than handcrafted vector approximations.
- P3: the live Android system navigation inset adds a small amount of bottom clearance that is absent from the artwork-only reference.

final result: passed
